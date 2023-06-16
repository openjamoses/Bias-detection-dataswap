from itertools import combinations

import numpy as np
import math
from scipy import integrate, stats, spatial
import torch
#from layers import SinkhornDistance
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)
def normal_pdf(x):
    return stats.norm.pdf(x) #math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))
def get_nonzero(val):
    val_= [i for i in val if i > 0]
    if len(val_) > 0:
        return min(val_)
    else:
        return min(val)

class Rankings:
    @staticmethod
    def rank_stability(r1, r2):
        '''SR(r, r′)= Summation(r-r1)^2/ (m)(m^2-1)'''
        stability = [(r1_i-r2_i)**2/len(r1)*((len(r1)**2)-1) for r1_i, r2_i in zip(r1, r2)]
        print('stability')
        return stability

def get_distance_summation(u, v):
    sum_euc_dist = 0
    n = len(u)
    pf_sum = 0
    list_distortion = []
    for i, j in combinations(range(len(x)), 2):
        pf_uv_original = x[j] - x[i]
        pf_uv_predict = y[j] - y[i]
        pf_uv = pf_uv_predict / pf_uv_original
        pf_sum += pf_uv

    return pf_sum

class Distances:
    @staticmethod
    def sigma_distance1(u, v):
        #u = normal_pdf(u)
        #v = normal_pdf(v)
        list_distances = []
        list_distances1 = []
        n = len(u)
        dict_x = {}
        dict_y = {}
        x = u
        y = v
        pf_sum = 0
        pf_sum_ = 0
        for i, j in combinations(range(len(x)), 2):
            pf=(y[i]-y[j])**2/((x[i]-x[j])**2)
            pf_sum += abs(pf)
            pf_sum_ += y[i]+y[j]+x[i]+x[j]

        for i, j in combinations(range(len(x)), 2):
            pf = abs((y[i] - y[j])**2 / (x[i] - x[j])**2)
            pf_ = (n * (n - 1) * pf) / (2 * pf_sum)  # n*(n-1)*
            pf_2 = (n * (n - 1) * pf) / (2 * pf_sum_)  # n*(n-1)*
            # list_distances.append((pf_-1)**2)
            list_distances.append((pf_-1)**2)
            list_distances1.append(pf_)
            print(x[i],x[j], y[i],y[j], pf_, pf)
        #print(' variance and mean: ', np.var(list_distances), np.mean(list_distances))
        return np.var(list_distances), np.mean(list_distances), np.var(list_distances1)

    @staticmethod
    def sigma_distance(u, v):

        print("\n\n*********\n")
        list_distances = []
        n = len(u)
        dict_x = {}
        dict_y = {}
        x = u
        y = v
        for i, j in combinations(range(len(x)), 2):
            #print(i, j)
            if i in dict_x.keys():
                dict_x[i].append([x[i], x[j]])
                dict_y[i].append([y[i], y[j]])
            else:
                dict_x[i] = [[x[i], x[j]]]
                dict_y[i] = [[y[i], y[j]]]
        pf_sum = 0
        for index, data in dict_x.items():
            euc_x = 0
            euc_y = 0
            for i in range(len(data)):
                dis = pow((data[i][0]-data[i][1]), 2)
                euc_x += dis

                #dis = pow((dict_y.get(index)[i][0] - dict_y.get(index)[i][1]), 2)
                euc_y += pow((dict_y.get(index)[i][0] - dict_y.get(index)[i][1]), 2)
            pf_sum += math.sqrt(euc_x)/math.sqrt(euc_y)
        for index, data in dict_x.items():
            euc_x = 0
            euc_y = 0
            for i in range(len(data)):
                dis = pow((data[i][0]-data[i][1]), 2)
                euc_x += dis
                euc_y += pow((dict_y.get(index)[i][0] - dict_y.get(index)[i][1]), 2)

                #dis = pow((dict_y.get(index)[i][0] - dict_y.get(index)[i][1]), 2)
            pf = math.sqrt(euc_x)/math.sqrt(euc_y)
            pf_ = (n*(n-1)*pf)/(2*pf_sum) #n*(n-1)*
            #list_distances.append((pf_-1)**2)
            list_distances.append((pf_-1)**2)
            print(index, pf_, pf, (pf_-1)**2)
        return np.var(list_distances), np.mean(list_distances)
def f(z, μ, Σ):
    print('z', z)
    #detZ = np.linalg.det(z)
    #print('detZ', detZ)
    #μ = np.mean(z)
    #print(z.shape, μ.shape, Σ.shape)
    """
    The density function of multivariate normal distribution.

    Parameters
    ---------------
    z: ndarray(float, dim=2)
        random vector, N by 1
    μ: ndarray(float, dim=1 or 2)
        the mean of z, N by 1
    Σ: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1
    """

    z = np.atleast_2d(z)
    μ = np.atleast_2d(μ)
    Σ = np.atleast_2d(Σ)

    N = z.size

    #print(z)
    det_ = np.linalg.det(Σ)
    temp1 = det_ ** (-1/2)
    #print('inverse matrix: ', np.linalg.det(Σ))

    if det_ != 0:
        temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))
    else:
        temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.pinv(Σ) @ (z - μ))
    #temp2 = np.exp(-.5 * (z - μ).T @ np.linalg.inv(Σ) @ (z - μ))

    return (2 * np.pi) ** (-N/2) * temp1 * temp2


class DistanceMeasure:
    @staticmethod
    def hellinger_discrete_1d(p, q):
        p = [x / np.sum(p) for x in p]
        q = [x / np.sum(q) for x in q]
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        """Hellinger distance between two discrete distributions.
           In pure Python.
           Some improvements.
        """
        return math.sqrt(sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)

    @staticmethod
    def hellinger_continous_1d(p, q):
        p = normal_pdf(p)  # [x/np.sum(p) for x in p]
        q = normal_pdf(q)  # [x / np.sum(q) for x in q]
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))#.pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q)) #stats.uniform.pdf(q)
        return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))

    @staticmethod
    def hellinger_multivariate(X, Y):
        """ Calculates Hellinger distance between 2 multivariate normal distribution
             X = X(x1, x2)
             Y = Y(y1, y2)
             The definition can be found at https://en.wikipedia.org/wiki/Hellinger_distance
        """
        if len(X) < 2 or len(Y) < 2:      return 1.

        meanX = np.mean(X, axis=0)
        covX = np.cov(X, rowvar=0)
        detX = np.linalg.det(covX)

        meanY = np.mean(Y, axis=0)
        covY = np.cov(Y, rowvar=0)
        detY = np.linalg.det(covY)

        detXY = np.linalg.det((covX + covY) / 2)
        if (np.linalg.det(covX + covY) / 2) != 0:
            covXY_inverted = np.linalg.inv((covX + covY) / 2)
        else:
            covXY_inverted = np.linalg.pinv((covX + covY) / 2)
        dist = 1. - (detX ** .25 * detY ** .25 / detXY ** .5) * np.exp(
            -.125 * np.dot(np.dot(np.transpose(meanX - meanY), covXY_inverted), (meanX - meanY)))

        # dist_2 = 1/np.sqrt(2)*np.linalg.norm(np.sqrt(detX)-np.sqrt(detY))

        # dis_3 = DistanceMeasure.hellinger_discrete_1d(X,Y)
        # dis_4 = DistanceMeasure.hellinger_continous_1d(X, Y)
        # print('hellinger_multivariate: ', dist, dist_2,dis_3, dis_4)
        # print('hellinger_multivariate: ', meanX, meanY, covXY_inverted)'''
        # X, Y = np.array(X).flatten(), np.array(Y).flatten()
        # dis_3 = DistanceMeasure.hellinger_discrete_1d(X, Y)
        # return dist

        return min(max(dist, 0.), 1.)

    @staticmethod
    def js_divergence(p, q):
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p), min(q))
        b = max(max(p), max(q))

        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        # pdf_normalP = normal_pdf(p, np.mean(p), np.std(p))
        # pdf_normalQ = normal_pdf(q, np.mean(q), np.std(q))
        # kl_diverg1 = stats.entropy(pdf_normalP, pdf_normalQ, base=2)

        # print('pdf_normalQ: ', pdf_normalP, pdf_normalQ)

        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)

        # print('KL compare: ', kl_diverg1, kl_diverg)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        # js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy

    @staticmethod
    def js_divergence_2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p.flatten()), min(q.flatten()))
        b = max(max(p.flatten()), max(q.flatten()))
        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        meanp = np.mean(p)  # , axis=0
        covp = np.cov(p)  # , rowvar=0
        # detX = np.linalg.det(covX)

        meanq = np.mean(q)  # , axis=0
        covq = np.cov(q)  # , rowvar=0
        # detY = np.linalg.det(covY)

        p_dist = f(p, meanp, covp)
        q_dist = f(q, meanq, covq)

        print('p_dist: ', p_dist)
        print('q_dist: ', q_dist)

        kl_diverg = stats.entropy(p_dist, q_dist, base=2)

        print('kl diverg manual: ', np.mean(kl_diverg))

        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)

        print('kl diverg Automated: ', kl_diverg)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        # js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy

    @staticmethod
    def wasserstein_distance(p, q):
        print('p before: ', p)
        p = np.array(p)
        q = np.array(q)
        p = p.flatten()
        q = q.flatten()
        print('p before: ', p)
        p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            dis_ = stats.wasserstein_distance(p, q)
        except Exception as e:
            dis_ = -1
        return dis_

    @staticmethod
    def wasserstein_distance_pdf(p, q):

        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p), min(q))
        b = max(max(p), max(q))

        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        print(' Total pdf: ', np.sum(Y1), np.sum(Y2))

        # print('p before: ', p)
        # p = np.array(p)
        # q = np.array(q)
        # p = p.flatten()
        # q = q.flatten()
        # print('p before: ', p)
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            p = [x / np.sum(p) for x in p]
            q = [x / np.sum(q) for x in q]

            dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
            dis_ = dis_ * 0.5
            # dis_ = stats.wasserstein_distance(Y1, Y2)

        except Exception as e:
            print('Error in wasserstein_distance_pdf: ', e)
            dis_ = -1
        return dis_

    @staticmethod
    def total_variation_distance(p, q):
        p = [x / np.sum(p) for x in p]
        q = [x / np.sum(q) for x in q]
        dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
        dis_ = dis_ * 0.5
        return dis_

    @staticmethod
    def wasserstein_distance_pdf2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p.flatten()), min(q.flatten()))
        b = max(max(p.flatten()), max(q.flatten()))
        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        Y1 = hist1_dist.pdf(X)
        Y2 = hist2_dist.pdf(X)

        # print('p before: ', p)
        # p = np.array(p)
        # q = np.array(q)
        # p = p.flatten()
        # q = q.flatten()
        # print('p before: ', p)
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            # p = p.flatten()
            # q = q.flatten()
            # p = [x / np.sum(p) for x in p]
            # q = [x / np.sum(q) for x in q]

            dis_ = stats.wasserstein_distance(Y1, Y2)
        except Exception as e:
            print('Error in wasserstein_distance_pdf: ', e)
            dis_ = -1
        return dis_

    @staticmethod
    def cramers_v(p, q):
        if len(p) > 0:
            # print('error seems to be here 0..!')
            vals, count = stats.contingency.crosstab(p, q)

            # data_crosstab = pd.crosstab(p, q, margins=False)
            # print('error seems to be here 0 end..!')
            # print('count:', count, vals)
            # print('data_crosstab: ', data_crosstab)

            # print('error seems to be here 1..!')
            # assoc_1 = stats.contingency.association(count, method="cramer")
            assoc_2 = stats.contingency.association(count, method="pearson")
            # print('error seems to be here 1 ends..!')

            # create 2x2 table
            # data = count #np.array([p, q])

            # Chi-squared test statistic, sample size, and minimum of rows and columns
            # print('error seems to be here 2..!')
            # X2 = stats.chi2_contingency(data, correction=False)[0]
            # print('error seems to be here 2 end..!')
            # n = np.sum(data)
            # minDim = min(data.shape) - 1

            # calculate Cramer's V
            # print('error seems to be here 3..!')
            # V = np.sqrt((X2 / n) / minDim)
            # print('error seems to be here 3 ends..!')

            # print('chi2_contingency: ', assoc_1, assoc_2, V)

            # display Cramer's V
            # print(V)
            # print(' --- coef: ', assoc_1, assoc_2)
            return assoc_2
        else:
            return -1
class DistanceMeasure_specific:
    @staticmethod
    def fair_degree(p, q, alpha=0.01):
        ydiff = [abs(a - b) for a, b in zip(p, q)]
        _ydiff = [1 / (abs(a - b)) for a, b in zip(p, q) if abs(a - b) > 0]
        if len(ydiff) == 0:
            ydiff.append(0)

        if len(_ydiff) == 0:
            _ydiff.append(0)
        perc_ydiff = np.percentile(ydiff, (1 - alpha) * 100)
        perc_ydiff_ = np.percentile(_ydiff, (1 - alpha) * 100)
        ydiff_contrained = [a for a in ydiff if a <= perc_ydiff]
        _ydiff_contrained = [a for a in _ydiff if a <= perc_ydiff_]
        wc_ = max(ydiff) * max(_ydiff)
        wc = max(ydiff_contrained) * max(_ydiff_contrained)
        # print('Worse case distance: ', wc_, wc)
        # print('ydiff: ', ydiff_contrained, _ydiff_contrained)
        # print('ydiff: ', sorted(ydiff), sorted(_ydiff))
        # for i, j in combinations(range(len(p)), 2):

        return wc_, wc
    @staticmethod
    def effect_distance(p,q, alpha=0.01):
        ydiff = [abs(a - b) for a, b in zip(p, q)]
        _ydiff = [1 / (abs(a - b)) for a, b in zip(p, q) if abs(a - b) > 0]
        if len(_ydiff) == 0:
            _ydiff.append(0)
        if len(ydiff) == 0:
            ydiff.append(0)
        return np.mean(ydiff), np.mean(_ydiff)

    @staticmethod
    def _hamming_distance(p,q):
        list_val = []
        for i in range(len(p)):
            distance = abs(p[i]-q[i])
            list_val.append(distance)
        return np.sum(list_val)/len(list_val)
    @staticmethod
    def _quared_error_proportion(p, q):
        data_p = {}
        data_q = {}
        distance_list = []
        for i, j in combinations(range(len(p)), 2):
            if i in data_p.keys():
                data_p[i].append([p[i], p[j]])
                data_q[i].append([q[i], q[j]])
            else:
                data_p[i] = [[p[i], p[j]]]
                data_q[i] = [[q[i], q[j]]]
        sum_pf = 0
        for index, data_ in data_p.items():
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0] - val[1]) ** 2
            for val in data_q.get(index):
                euc_q += (val[0] - val[1]) ** 2
            pf = 0
            if euc_q != euc_p:
                if euc_p != 0:
                    pf = math.sqrt(euc_q) / math.sqrt(euc_p)
            sum_pf += pf

        for index, data_ in data_p.items():
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0] - val[1]) ** 2
            for val in data_q.get(index):
                euc_q += (val[0] - val[1]) ** 2
            pf = 0
            if euc_q != euc_p:
                if euc_p != 0:
                    pf = math.sqrt(euc_q) / math.sqrt(euc_p)
            #pf_ = pf/sum_pf
            if sum_pf > 0:
                pf_ = pf / sum_pf
            else:
                pf_ = pf
            distance_list.append(pf_)
        return np.mean(distance_list)
    @staticmethod
    def _distance_single(p, q, discrete_indices=[]):
        distance_list = []
        list_index = []
        #print('distortion single starts ----')
        for i in range(len(p)):
            distance = 0
            if i in discrete_indices:
                if p[i] != q[i]:
                    distance = 1
                distance_list.append(distance)
            else:
                list_index.append(i)

        data_p = {}
        data_q = {}
        for i, j in combinations(list_index, 2):
            if i in data_p.keys():
                data_p[i].append([p[i], p[j]])
                data_q[i].append([q[i], q[j]])
            else:
                data_p[i] = [[p[i], p[j]]]
                data_q[i] = [[q[i], q[j]]]


        for index, data_ in data_p.items():
            euc_p = 0
            euc_q = 0
            for val in data_:
                euc_p += (val[0] - val[1]) ** 2
            for val in data_q.get(index):
                euc_q += (val[0] - val[1]) ** 2
            pf = 0
            if euc_q != euc_p:
                if euc_p != 0:
                    pf = math.sqrt(euc_q) / math.sqrt(euc_p)
                pf = abs(pf - 1)

            distance_list.append(pf)
            #print('distortion single ends ----')
        return np.mean(distance_list)

    @staticmethod
    def _distance_multiple(p, q, discrete_indices=[], alpha=0.25):
        list_val  = []
        for a in range(len(p)):
            p_a = p[a]
            q_a = q[a]
            distance_list = []
            list_index = []
            for i in range(len(p_a)):
                distance = 0
                if i in discrete_indices:
                    if p_a[i] != q_a[i]:
                        distance = 1
                    distance_list.append(distance)
                else:
                    list_index.append(i)

            data_p = {}
            data_q = {}
            for i, j in combinations(list_index, 2):
                if i in data_p.keys():
                    data_p[i].append([p_a[i], p_a[j]])
                    data_q[i].append([q_a[i], q_a[j]])
                else:
                    data_p[i] = [[p_a[i], p_a[j]]]
                    data_q[i] = [[q_a[i], q_a[j]]]

            for index, data_ in data_p.items():
                euc_p = 0
                euc_q = 0
                for val in data_:
                    euc_p += (val[0] - val[1]) ** 2
                for val in data_q.get(index):
                    euc_q += (val[0] - val[1]) ** 2
                pf = 0
                if euc_q != euc_p:
                    if euc_p != 0:
                        pf = math.sqrt(euc_q) / math.sqrt(euc_p)
                    pf = abs(pf - 1)

                distance_list.append(pf)
            if alpha != None:
                if np.mean(distance_list) <= alpha:
                    list_val.append(np.mean(distance_list))
                else:
                    print('  --- distance greater than alpha: ', np.mean(distance_list))
        return np.mean(list_val)

    @staticmethod
    def hellinger_discrete_1d(p, q):
        #p = [x / np.sum(p) for x in p]
        #q = [x / np.sum(q) for x in q]
        p = normal_pdf(p)
        q = normal_pdf(q)
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        """Hellinger distance between two discrete distributions.
           In pure Python.
           Some improvements.
        """
        return math.sqrt(sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p, q)]) / 2)
    @staticmethod
    def hellinger_continous_1d(p,q):
        #p = [ x/np.sum(p) for x in p]
        #q = [x / np.sum(q) for x in q]

        p = normal_pdf(p)
        q = normal_pdf(q)
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))#.pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q)) #stats.uniform.pdf(q)
        return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(p)-np.sqrt(q))

    @staticmethod
    def hellinger_multivariate(X, Y):
        """ Calculates Hellinger distance between 2 multivariate normal distribution
             X = X(x1, x2)
             Y = Y(y1, y2)
             The definition can be found at https://en.wikipedia.org/wiki/Hellinger_distance
        """
        if len(X) < 2 or len(Y) < 2:      return 1.

        meanX = np.mean(X, axis=0)
        covX = np.cov(X, rowvar=0)
        detX = np.linalg.det(covX)

        meanY = np.mean(Y, axis=0)
        covY = np.cov(Y, rowvar=0)
        detY = np.linalg.det(covY)

        detXY = np.linalg.det((covX + covY) / 2)
        if (np.linalg.det(covX + covY) / 2) != 0:
            covXY_inverted = np.linalg.inv((covX + covY) / 2)
        else:
            covXY_inverted = np.linalg.pinv((covX + covY) / 2)
        dist = 1. - (detX ** .25 * detY ** .25 / detXY ** .5) * np.exp(
            -.125 * np.dot(np.dot(np.transpose(meanX - meanY), covXY_inverted), (meanX - meanY)))
        #print(dist)




        return min(max(dist, 0.), 1.)
    @staticmethod
    def js_divergence(p, q):
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0

        Y1 = normal_pdf(p)
        Y2 = normal_pdf(q)
        #fig, ax = plt.subplots(1, 1)
        #ax.hist(Y1, density=True, histtype='stepfilled', alpha=0.2)
        #ax.legend(loc='best', frameon=False)
        #plt.show()

        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)

        #kl_diverg2 = stats.entropy(pdf_normalP, pdf_normalQ, base=2)
        #print('KL divergence: ', kl_diverg, kl_diverg2)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        #js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy

    @staticmethod
    def js_divergence_2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0

        Y1 = normal_pdf(p) #hist1_dist.pdf(X)
        Y2 = normal_pdf(q) #hist2_dist.pdf(X)


        #print('p_dist: ', p_dist)
        #print('q_dist: ', q_dist)

        #kl_diverg =  min(stats.entropy(p_dist, q_dist, base=2))

        #print('kl diverg manual: ', kl_diverg)
        # compute KL divergence between Y1 and Y2
        kl_diverg = stats.entropy(Y1, Y2, base=2)
        kl_diverg = get_nonzero(kl_diverg)
        # compute KL divergence between Y1 and Y2
        #kl_diverg = stats.entropy(Y1, Y2, base=2)

        #print('kl diverg Automated: ', kl_diverg)

        # Obtain point-wise mean of the two PDFs Y1 and Y2, denote it as M
        M = (Y1 + Y2) / 2
        # Compute Kullback-Leibler divergence between Y1 and M
        d1 = stats.entropy(Y1, M, base=2)
        d1 = get_nonzero(d1)
        # d1 = 0.406
        # Compute Kullback-Leibler divergence between Y2 and M
        d2 = stats.entropy(Y2, M, base=2)
        d2 = get_nonzero(d2)
        # d2 = 0.300

        # Take the average of d1 and d2
        # we get the symmetric Jensen-Shanon divergence
        js_dv = (d1 + d2) / 2
        # js_dv = 0.353
        # Jensen-Shanon distance is the square root of the JS divergence
        js_distance = np.sqrt(js_dv)
        # js_distance = 0.594
        # Check it against scipy's calculation
        # js_distance_scipy = spatial.distance.jensenshannon(Y1, Y2)
        # js_distance_scipy = 0.493
        return kl_diverg, js_distance  # , js_distance_scipy
    @staticmethod
    def wasserstein_distance(p,q):
        #print('p before: ', p)
        p = np.array(p)
        q = np.array(q)
        #p = p.flatten()
        #q = q.flatten()
        #print('p before: ', p)
        #p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        #q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        #dis_ = stats.wasserstein_distance(p, q)
        try:
            #p = [x / np.sum(p) for x in p]
            #q = [x / np.sum(q) for x in q]
            p = normal_pdf(p)
            q = normal_pdf(q)
            dis_ = stats.wasserstein_distance(p,q)
        except Exception as e:
            dis_ = -1
        return dis_

    @staticmethod
    def wasserstein_distance_pdf_(p, q):
        #try:

        meanp = np.mean(p)  # , axis=0
        covp = np.cov(p)  # , rowvar=0
        # detX = np.linalg.det(covX)
        meanq = np.mean(q)  # , axis=0
        covq = np.cov(q)  # , rowvar=0
        # detY = np.linalg.det(covY)
        #p = f(p, meanp, covp)
        #q = f(q, meanq, covq)
        p = normal_pdf(p)  # hist1_dist.pdf(X)
        p = normal_pdf(q)  # hist2_dist.pdf(X)

        p = p.flatten()
        q = q.flatten()

        dis_ = stats.wasserstein_distance(p, q)

        #except Exception as e:
        #    #print('Error in wasserstein_distance_pdf: ', e)
        #    dis_ = -1
        return dis_
    @staticmethod
    def total_variation_distance(p,q):
        if np.sum(p) != 0:
            p = [x / np.sum(p) for x in p]
        if np.sum(q) != 0:
            q = [x / np.sum(q) for x in q]
        #p = normal_pdf(p)  # hist1_dist.pdf(X)
        #q = normal_pdf(q)  # hist2_dist.pdf(X)
        dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
        dis_ = dis_ * 0.5
        return dis_

    @staticmethod
    def total_variation_distance_2d(p, q):
        p = np.array(p).flatten()
        q = np.array(q).flatten()
        if np.sum(p) != 0:
            p = [x / np.sum(p) for x in p]
        if np.sum(q) != 0:
            q = [x / np.sum(q) for x in q]
        # p = normal_pdf(p)  # hist1_dist.pdf(X)
        # q = normal_pdf(q)  # hist2_dist.pdf(X)
        dis_ = sum([abs(p_i - q_i) for p_i, q_i in zip(p, q)])
        dis_ = dis_ * 0.5
        #print('total variation multivariate: ', )
        return dis_

    @staticmethod
    def wasserstein_distance_pdf2d(p, q):
        p = np.array(p)
        q = np.array(q)
        n1 = len(p)
        n2 = len(q)
        if n1 <= 1 or n2 <= 1:
            return 0.0, 0.0
        # Calculate the interval to be analyzed further
        a = min(min(p.flatten()), min(q.flatten()))
        b = max(max(p.flatten()), max(q.flatten()))
        # Plot the PDFs
        max_bins = max(n1, n2)
        # lin = linspace(a, b, max(n1, n2))
        # p = pdf1.pdf(lin)
        # q = pdf2.pdf(lin)
        # p = norm.pdf(p, loc=0, scale=1)
        # q = norm.pdf(q, loc=0, scale=1)

        # Construct empirical PDF with these two samples
        hist1 = np.histogram(p, bins=max_bins)  # bins=10
        hist1_dist = stats.rv_histogram(hist1)
        hist2 = np.histogram(q, bins=max_bins)  # bins=10
        hist2_dist = stats.rv_histogram(hist2)
        X = np.linspace(a, b, max_bins)
        #Y1 = hist1_dist.pdf(X)
        #Y2 = hist2_dist.pdf(X)

        p = p.flatten()
        q = q.flatten()

        Y1 = normal_pdf(p)  # hist1_dist.pdf(X)
        Y2 = normal_pdf(q)  # hist2_dist.pdf(X)



        # print('p before: ', p)
        # p = np.array(p)
        # q = np.array(q)
        # p = p.flatten()
        # q = q.flatten()
        # print('p before: ', p)
        # p = stats.norm.cdf(p, np.mean(p), np.std(p))  # .pdf(p)
        # q = stats.norm.cdf(q, np.mean(q), np.std(q))  # stats.uniform.pdf(q)
        # dis_ = stats.wasserstein_distance(p, q)
        try:
            #p = p.flatten()
            #q = q.flatten()
            #p = [x / np.sum(p) for x in p]
            #q = [x / np.sum(q) for x in q]

            dis_ = stats.wasserstein_distance(Y1, Y2)
        except Exception as e:
            print('Error in wasserstein_distance_pdf: ', e)
            dis_ = -1
        return dis_

    @staticmethod
    def cramers_v(p, q):
        if len(p) > 0:
            vals, count = stats.contingency.crosstab(p, q)

            # create 2x2 table
            data = count #np.array([p, q])

            # Chi-squared test statistic, sample size, and minimum of rows and columns
            X2 = stats.chi2_contingency(data, correction=False)[0]
            n = np.sum(data)
            minDim = min(data.shape) - 1

            # calculate Cramer's V
            V = np.sqrt((X2 / n) / minDim)

            # display Cramer's V
            #print(V)
            return V
        else:
            return -1

if __name__ == '__main__':

    x = [1,2,3,4,5,6,7,8,9,10]
    y = [3,2,0,4,5,1,7,6,2,1]

    print(combinations(x,2))

    sigma_distance = Distances.sigma_distance1(x,y)
    sigma_distance2 = Distances.sigma_distance(x, y)

    wasserstein_distance = DistanceMeasure_specific.wasserstein_distance(x,y)
    hellinger_distance = DistanceMeasure_specific.hellinger_continous_1d(x,y)
    total_variation_distance = DistanceMeasure_specific.total_variation_distance(x, y)
    js_divergence = DistanceMeasure_specific.js_divergence(x, y)
    cramers_v_distance = DistanceMeasure_specific.cramers_v(x, y)

    print('wasserstein_distance: ', wasserstein_distance)
    print('hellinger_distance: ', hellinger_distance)
    print('total_variation_distance: ',total_variation_distance)
    print('sigma_distance: ', sigma_distance)
    #print('sigma_distance2: ', sigma_distance2)




