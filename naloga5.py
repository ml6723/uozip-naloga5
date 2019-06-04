import numpy as np
import pandas as pd
import sys
from collections import Counter


def calculate_RMSE(predictions, real):
    return np.sqrt(np.mean([(predictions[x] - real[x]) ** 2 for x in range(0, len(real))]))


class RecommendationSystem:
    def __init__(self, train, K=50, alpha=0.001, lambda_=0):
        self.train = train

        # friends
        user_friends = np.genfromtxt("user_friends.dat", dtype=float, delimiter='\t', skip_header=1)
        self.user_friends_IDs = list(set(user_friends[:, 0]))

        self.friends = {x: [] for x in self.user_friends_IDs}

        for x in range(0, len(user_friends)):
            self.friends[user_friends[x, 0]] += [user_friends[x, 1]]

        self.user_count_scores = Counter(train[:, 0])
        self.artist_count_scores = Counter(train[:, 1])

        self.userIDs = list(set(self.train[:, 0]))
        self.artistIDs = list(set(self.train[:, 1]))

        print(len(self.userIDs))
        print(len(self.artistIDs))

        self.user_indexes = {self.userIDs[x]: x for x in range(0, len(self.userIDs))}
        self.artist_indexes = {self.artistIDs[x]: x for x in range(0, len(self.artistIDs))}

        self.R = np.zeros((len(self.userIDs), len(self.artistIDs)))
        self.R_train = np.zeros((len(self.userIDs), len(self.artistIDs)))
        self.R_validation = np.zeros((len(self.userIDs), len(self.artistIDs)))

        self.avg_score = np.mean(self.train[:, 2])

        for i in range(0, len(self.train)):
            x = self.user_indexes[self.train[i, 0]]
            y = self.artist_indexes[self.train[i, 1]]

            self.R[x, y] = self.train[i, 2]

        self.user_bias = {}  # keys are user_indexes for user
        self.user_avg = {}
        self.avg_users = np.true_divide(self.R.sum(1), (self.R != 0).sum(1))
        for i in self.userIDs:
            self.user_bias[self.user_indexes[i]] = self.avg_users[self.user_indexes[i]] - self.avg_score
            self.user_avg[self.user_indexes[i]] = self.avg_users[self.user_indexes[i]]

        self.artist_bias = {}  # keys are artist_indexes for artist
        self.artist_avg = {}
        self.avg_artists = np.true_divide(self.R.sum(0), (self.R != 0).sum(0))
        for i in self.artistIDs:
            self.artist_bias[self.artist_indexes[i]] = self.avg_artists[self.artist_indexes[i]] - self.avg_score
            self.artist_avg[self.artist_indexes[i]] = self.avg_artists[self.artist_indexes[i]]

        for i in range(0, len(self.train)):
            x = self.user_indexes[self.train[i, 0]]
            y = self.artist_indexes[self.train[i, 1]]

            rand = np.random.random()

            if rand <= 0.3:
                self.R_validation[x, y] = self.train[i, 2]
            else:
                self.R_train[x, y] = self.train[i, 2]

        self.P = np.random.random_sample((len(self.userIDs), K)) * 0.001
        self.Q = np.random.random_sample((K, len(self.artistIDs))) * 0.001

        self.user_bias_vector = np.array(list(self.user_bias.values()))
        self.artist_bias_vector = np.array(list(self.artist_bias.values()))

        # self.user_bias_vector = np.zeros(len(self.userIDs))
        # self.artist_bias_vector = np.zeros(len(self.artistIDs))

        # self.P[:,0] = 1
        # self.Q[1,:] = 1
        #
        # self.P[:,1] = self.user_bias_vector
        # self.Q[0,:] = self.artist_bias_vector

        rms_err_prev_prev = np.finfo(np.float64).max
        rms_err_prev = rms_err_prev_prev / 2
        rms_err = rms_err_prev / 2

        self.P_prev = np.array([0])
        self.Q_prev = np.array([0])

        while True:
            print(rms_err)

            if (rms_err_prev < rms_err) & (rms_err_prev_prev < rms_err_prev):
                print(rms_err_prev_prev)

                self.P = self.P_prev_prev
                self.Q = self.Q_prev_prev

                break

            self.P_prev_prev = np.copy(self.P_prev)
            self.Q_prev_prev = np.copy(self.Q_prev)

            self.P_prev = np.copy(self.P)
            self.Q_prev = np.copy(self.Q)

            rms_err_prev_prev = rms_err_prev
            rms_err_prev = rms_err

            for x in range(0, len(self.train)):
                u = self.user_indexes[self.train[x, 0]]
                i = self.artist_indexes[self.train[x, 1]]

                if self.R_train[u, i] == 0:
                    continue

                p_u = self.P[u, :]
                q_i = self.Q[:, i]

                e_ui = (self.R_train[u, i] - (p_u.dot(q_i)) + self.user_bias_vector[u] + self.artist_bias_vector[
                    i] + self.avg_score) + lambda_ * (p_u.dot(p_u) + q_i.dot(q_i))

                self.P[u, :] = (p_u + alpha * (e_ui * q_i - lambda_ * p_u))
                self.Q[:, i] = (q_i + alpha * (e_ui * p_u - lambda_ * q_i))

                # self.user_bias_vector[u] += alpha * (e_ui - lambda_ * self.user_bias_vector[u])
                # self.artist_bias_vector[i] += alpha * (e_ui - lambda_ * self.artist_bias_vector[i])

                # self.P[u,0] = 1
                # self.Q[1, i] = 1

                # self.P[:, 1] = self.user_bias_vector
                # self.Q[0, :] = self.artist_bias_vector

            x, y = np.where(self.R_validation)

            predictions = self.P.dot(self.Q)

            rms_err = np.sqrt(np.mean([((
                                        predictions[x[i], y[i]] + self.user_bias_vector[x[i]] + self.artist_bias_vector[
                                            y[i]] + self.avg_score) - self.R_validation[x[i], y[i]]) ** 2 for i in range(0, len(x))]))

    def predict(self, u, i):
        u_index = self.user_indexes.get(u)
        i_index = self.artist_indexes.get(i)

        if (u_index == None) & (i_index != None):
            # if self.friends.get(u) != None:
            #     friends_predictions = [self.P[self.user_indexes[x],:].dot(self.Q[:, i_index]) for x in self.friends[u] if self.user_indexes.get(x) != None]
            #     if friends_predictions != []:
            #         prediction = np.mean(friends_predictions)
            #     else:
            #         prediction = self.Q[0, i_index]
            # else:
            #     prediction = self.Q[0, i_index]

            # prediction = self.Q[0, i_index] + self.avg_score

            prediction = self.artist_bias_vector[i_index] + self.avg_score

            # prediction = self.artist_avg[i_index] #+ self.avg_score
        elif (i_index == None) & (u_index != None):
            # prediction = self.P[u_index, 1] + self.avg_score

            prediction = self.user_bias_vector[u_index] + self.avg_score

            # prediction = self.user_avg[u_index] #+ self.avg_score
        elif (u_index == None) & (i_index == None):
            # if self.friends.get(u) != None:
            #     friends_predictions = [self.P[self.user_indexes[x], 1] for x in self.friends[u] if self.user_indexes.get(x) != None]
            #     if friends_predictions != []:
            #         prediction = np.mean(friends_predictions)
            #     else:
            #         prediction = self.avg_score
            # else:
            #     prediction = self.avg_score

            prediction = self.avg_score
        else:
            p_u = self.P[u_index, :]
            q_i = self.Q[:, i_index]

            prediction = p_u.dot(q_i) + self.user_bias_vector[u_index] + self.artist_bias_vector[i_index] + self.avg_score

        if prediction < 0:
            prediction = 0

        return prediction


data = np.genfromtxt("user_artists_training.dat", dtype=float, delimiter='\t')
test = np.genfromtxt("user_artists_test.dat", dtype=float, delimiter='\t')

# # internal validation 5x
# RMSEs = []
#
# for i in range(0, 5):
#     data_train = np.copy(data[1:, :])
#     data_test = np.copy(data_train)
#
#     indexes = [np.random.random() for x in range(0, len(data_train))]
#
#     for x in range(0, len(indexes)):
#         if indexes[x] <= 0.3:
#             data_train = np.delete(data_train, x, 0)
#         else:
#             data_test = np.delete(data_test, x, 0)
#
#     rs = RecommendationSystem(data_train)
#
#     predictions = [rs.predict(data_test[x, 0], data_test[x, 1]) for x in range(0, len(data_test))]
#
#     rms_err = calculate_RMSE(predictions, data_test[:,2])
#
#     print('RMSE:')
#     print(rms_err)
#     RMSEs.append(rms_err)
#
# print('avg RMSE:')
# print(np.mean(RMSEs))
#
# exit()



# internal validation 1x
data_train = np.copy(data[1:, :])
data_test = np.copy(data_train)

indexes = [np.random.random() for x in range(0, len(data_train))]

deleted_train = 0
deleted_test = 0

for x in range(0, len(indexes)):
    if indexes[x] <= 0.3:
        data_train = np.delete(data_train, x - deleted_train, 0)
        deleted_train += 1
    else:
        data_test = np.delete(data_test, x - deleted_test, 0)
        deleted_test += 1

rs = RecommendationSystem(data_train)

predictions = [rs.predict(data_test[x, 0], data_test[x, 1]) for x in range(0, len(data_test))]

rms_err = calculate_RMSE(predictions, data_test[:, 2])

print(rms_err)

exit()

# # recommendation for me as an user
# data = np.genfromtxt("user_artists_training_me.dat", dtype=float, delimiter='\t')
# artists = pd.read_csv("artists.dat", delimiter='\t').as_matrix()
#
# rs = RecommendationSystem(data[1:, :])
#
# predictions = {}
# for x in range(0, len(artists)):
#     artist_ID = float(artists[x, 0])
#     artist_name = artists[x, 1]
#     user = data[-1,0]
#
#     predictions[artist_name] = rs.predict(user, artist_ID)
#
# predictions_sorted = sorted(predictions.items() , key=lambda v : v[1], reverse=True)
#
# f = open('my_predictions.txt', 'w')
#
# for k,v in predictions_sorted:
#     print(k.encode("utf-8"),v, file=f)
#
# f.close()
#
# i = 0
# for k,v in predictions_sorted:
#     if i == 10:
#         break
#     print(k.encode("utf-8"), v)
#     i += 1
#
# exit()



rs = RecommendationSystem(data[1:, :])

f = open('results_pred.txt', 'w')

for x in range(1, len(test)):
    u = test[x, 0]
    i = test[x, 1]

    print(rs.predict(u, i), file=f)

f.close()
