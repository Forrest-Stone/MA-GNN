from model.gating_network import MAGNN
from interactions import Interactions
from eval_metrics import *

import argparse
import logging
from time import time
import datetime
import torch

logging.basicConfig(level=logging.DEBUG,
                    filename='magnn-test.log', filemode='a')
logger = logging.getLogger(__name__)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def evaluation(magnn, train, test_set, topk=20):
    num_users = train.num_users
    num_items = train.num_items
    # batch_size = 1024
    batch_size = 256
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    train_matrix = train.tocsr()
    test_sequences = train.test_sequences.sequences
    test_left_sequences = train.test_sequences.left_sequence

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)
        batch_test_left_sequences = test_left_sequences[batch_user_index]
        batch_test_left_sequences = np.atleast_2d(batch_test_left_sequences)

        gnn_np = torch.Tensor(
            get_slice(batch_test_sequences, True)).float().to(device)

        batch_test_sequences = torch.from_numpy(
            batch_test_sequences).type(torch.LongTensor).to(device)
        batch_test_left_sequences = torch.from_numpy(
            batch_test_left_sequences).type(torch.LongTensor).to(device)

        item_ids = torch.from_numpy(item_indexes).type(
            torch.LongTensor).to(device)
        batch_user_ids = torch.from_numpy(
            np.array(batch_user_index)).type(torch.LongTensor).to(device)

        # 测试的时候这个地方内存过大报错
        with torch.no_grad():
            rating_pred = magnn(batch_test_sequences, batch_test_left_sequences,
                                batch_user_ids, item_ids, gnn_np, True)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(
            arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[
                              :, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def generate_negative_samples(train_matrix, num_neg=3, num_sets=10):
    neg_samples = []
    for user_id, row in enumerate(train_matrix):
        pos_ind = row.indices
        neg_sample = negsamp_vectorized_bsearch_preverif(
            pos_ind, train_matrix.shape[1], num_neg * num_sets)
        neg_samples.append(neg_sample)

    return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def get_slice(inputs_np, for_pre=False):
    inputs = inputs_np
    n_node, A = [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        if for_pre:
            u_A = np.array([[0, 1/3, 1/3, 1/3, 0, 0], [1/4, 0, 1/4, 1/4, 1/4, 0], [1/5, 1/5, 0, 1/5, 1/5, 1/5], [
                           1/5, 1/5, 1/5, 0, 1/5, 1/5], [0, 1/4, 1/4, 1/4, 0, 1/4], [0, 0, 1/3, 1/3, 1/3, 0]])
        else:
            u_A = np.array([[0, 1/3, 1/3, 1/3, 0, 0], [1/4, 0, 1/4, 1/4, 1/4, 0], [1/5, 1/5, 0, 1/5, 1/5, 1/5], [
                           1/5, 1/5, 1/5, 0, 1/5, 1/5], [0, 1/4, 1/4, 1/4, 0, 1/4], [0, 0, 1/3, 1/3, 1/3, 0]])
        A.append(u_A)
    return A


def train_model(train_data, test_data, config):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    left_sequence_np = train_data.sequences.left_sequence

    # # generate train sequence
    # sequences_np = np.concatenate((sequences_np, targets_np), axis=1)

    users_np = train_data.sequences.user_ids
    train_matrix = train_data.tocsr()

    left_number = left_sequence_np.shape[1]
    # print(left_number)
    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))

    magnn = MAGNN(num_users, num_items, config, left_number, device).to(device)

    optimizer = torch.optim.Adam(
        magnn.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    t0 = time()

    for epoch_num in range(config.n_iter):

        t1 = time()

        # set model to training mode
        magnn.train()

        np.random.shuffle(record_indexes)

        t_neg_start = time()
        negatives_np_multi = generate_negative_samples(
            train_matrix, config.neg_samples, config.sets_of_neg_samples)
        logger.info("Negative sampling time: {}s".format(time() - t_neg_start))

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_left_sequences = left_sequence_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            negatives_np = negatives_np_multi[batchID %
                                              config.sets_of_neg_samples]
            batch_neg = negatives_np[batch_users]

            gnn_np = torch.Tensor(
                get_slice(batch_sequences, False)).float().to(device)

            batch_users = torch.from_numpy(batch_users).type(
                torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(
                batch_sequences).type(torch.LongTensor).to(device)
            batch_left_sequences = torch.from_numpy(
                batch_left_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(
                batch_targets).type(torch.LongTensor).to(device)
            batch_negatives = torch.from_numpy(
                batch_neg).type(torch.LongTensor).to(device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

            prediction_score = magnn(
                batch_sequences, batch_left_sequences, batch_users, items_to_predict, gnn_np, False)

            (targets_prediction, negatives_prediction) = torch.split(
                prediction_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

            # compute the BPR loss
            loss = -torch.log(torch.sigmoid(targets_prediction -
                                            negatives_prediction) + 1e-8)
            loss = torch.mean(torch.sum(loss))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= num_batches

        t2 = time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (
            epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        # with torch.no_grad():
        # 输出评价指标
        magnn.eval()
        precision, recall, MAP, ndcg = evaluation(
            magnn, train_data, test_data, topk=20)

        if (epoch_num + 1) % 20 == 0:
            # magnn.eval()
            # precision, recall, MAP, ndcg = evaluation(
            #     magnn, train_data, test_data, topk=20)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in MAP))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))

    logger.info("\n")
    logger.info("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=6)
    parser.add_argument('--T', type=int, default=2)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=2)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    parser.add_argument('--step', type=int, default=2,
                        help='gnn propogation steps')
    parser.add_argument('--h', type=int, default=20,
                        help='number of dimensions in attention')
    parser.add_argument('--m', type=int, default=20,
                        help='number of memory units')
    parser.add_argument('--dataset', default='CDs',
                        help='the datasets of model')

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50,
                        help='item embedding size')
    # parser.add_argument('--d2', type=int, default=128,
    #                     help='user embedding size')

    config = parser.parse_args()

    from data import Amazon
    from data import GoodReads
    from data import MovieLens

    if config.dataset == 'CDs':
        data_set = Amazon.CDs()
    elif config.dataset == 'Books':
        data_set = Amazon.Books()
    elif config.dataset == 'Comics':
        data_set = GoodReads.Comics()
    elif config.dataset == 'Children':
        data_set = GoodReads.Children()
    else:
        data_set = MovieLens.ML20M()

    # item_id=0 for sequence padding
    config.dataset = 'CDs'
    data_set = Amazon.CDs()
    train_set, val_set, train_val_set, test_set, num_users, num_items = data_set.generate_dataset(
        index_shift=1)
    train = Interactions(train_val_set, num_users, num_items)
    train.to_sequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)
    train_model(train, test_set, config)
