# -*- coding:utf-8 -*-

import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    # print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import multiprocessing
import matplotlib.pyplot as plt
font_size=14


def random_pick_probability(numbers,probabilities):
    prob =  np.random.uniform(0,1,len(numbers))
    select = np.array(probabilities) - prob
    # print "select:",select>0
    return np.array(numbers)[select>0]

# np.random.seed(100)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': '{: 0.10f}'.format})




####################################################################
# experiment for private budget
# sample and aggreation mechainsm
####################################################################
def sample_aggreation(data,epsilon = 1.0,delta = 0.00001,number_of_partition = 5,beta = 0.5):
    # number_of_partition = 5
    # beta = 0.5
    # epsilon = 1.0
    # delta = 0.00001
    alpha_1 = np.true_divide(epsilon, 2)
    alhpa_2 = np.true_divide(epsilon, 2) * np.log(np.true_divide(1, delta))


    z_array = np.array([0.0 for a in np.arange(0, number_of_partition, 1)])
    for loop in np.arange(0, number_of_partition, 1):
        sampeld_data = random_pick_probability(data, 0.5)
        # print "data and lenght:",len(data),":::",data
        # print "sampled_data and lenght:",len(sampeld_data),":::",sampeld_data
        z_array[loop] = np.average(sampeld_data)

    z_array = sorted(z_array)
    # print "z_array:", z_array
    k = np.floor(np.sqrt(number_of_partition))
    # print "k:", k
    slected_number_1 = z_array[len(z_array) - int(k):]
    slected_number_2 = z_array[0:int(k)]
    # print "slected_number_1:", slected_number_1
    # print "slected_number_2:", slected_number_2
    if np.abs(np.sum(slected_number_1)) > np.abs(np.sum(slected_number_2)):
        slected_number = slected_number_1
    else:
        slected_number = slected_number_2
    # print "slected number:", slected_number
    local_sensitivity = np.average(z_array) - (np.sum(z_array) - np.sum(slected_number)) / (len(z_array) - k)
    # print "local_sensitivity:", local_sensitivity
    smooth_sensitivity = local_sensitivity * np.exp(beta)
    # print "smooth_sensitivity:", smooth_sensitivity

    real_result = np.average(data)
    noise_result = np.average(z_array) + np.true_divide(local_sensitivity, alpha_1) * np.random.laplace(0, 1)
    # print "real_result:", real_result
    # print "noise_result:", noise_result
    return noise_result,real_result

# data = np.random.normal(mean, standard_deviation, number_of_number)
# print "result of function:",sample_aggreation(data)
####################################################################

####################################################################
# experiment for private budget
# personlized sample and aggreation mechainsm
# number of sampel = 5
####################################################################

def personalized_sample_aggreation(data,epsilon = 1.0,delta = 0.00001,number_of_partition = 5,beta = 0.5,constant=2):
    alpha_1 = np.true_divide(epsilon, 2)
    alhpa_2 = np.true_divide(epsilon, 2) * np.log(np.true_divide(1, delta))

    # z_array = np.array([0.0 for a in np.arange(0, number_of_partition, 1)])

    record_sensitivity = abs(np.true_divide(data, len(data)))

    sample_probability = [0 for a in record_sensitivity]
    for round in np.arange(0, len(record_sensitivity), 1):
        if record_sensitivity[round] > constant:
            sample_probability[round] = np.true_divide((1 - np.exp(epsilon)), (
                    1 - np.exp(epsilon * np.true_divide(record_sensitivity[round], constant))))
        else:
            sample_probability[round] = 1
    sample_probability = np.true_divide(sample_probability,2.0)
    z_array = np.array([0.0 for a in np.arange(0, number_of_partition, 1)])
    for loop in np.arange(0, number_of_partition, 1):
        sampled_data = random_pick_probability(data, sample_probability)
        # print "data and lenght:",len(data),":::",data
        # print "sampled_data and lenght:",len(sampeld_data),":::",sampeld_data
        while len(sampled_data) == 0:
            print "the sampled data is empty!!!"
            sampled_data = random_pick_probability(data, sample_probability)
        z_array[loop] = np.average(sampled_data)

    z_array = sorted(z_array)
    # print "z_array:", z_array
    k = np.floor(np.sqrt(number_of_partition))
    # print "k:", k
    slected_number_1 = z_array[len(z_array) - int(k):]
    slected_number_2 = z_array[0:int(k)]
    # print "slected_number_1:", slected_number_1
    # print "slected_number_2:", slected_number_2
    if np.abs(np.sum(slected_number_1)) > np.abs(np.sum(slected_number_2)):
        slected_number = slected_number_1
    else:
        slected_number = slected_number_2
    # print "slected number:", slected_number
    local_sensitivity = np.average(z_array) - (np.sum(z_array) - np.sum(slected_number)) / (len(z_array) - k)
    # print "local_sensitivity:", local_sensitivity
    smooth_sensitivity = local_sensitivity * np.exp(beta)
    # print "smooth_sensitivity:", smooth_sensitivity

    real_result = np.average(data)
    noise_result = np.average(z_array) + np.true_divide(local_sensitivity, alpha_1) * np.random.laplace(0, 1)
    # print "real_result:", real_result
    # print "noise_result:", noise_result
    return noise_result, real_result

####################################################################
# experiment for private budget
# personlized sample and aggreation mechainsm
# number of sampel = 1
####################################################################

def personlized_sample(data, epsilon=1.0, delta=0.00001, number_of_partition=1, beta=0.5, constant=2.0):
    # number_of_partition = 5
    # beta = 0.5
    # epsilon = 1.0
    # delta = 0.00001
    alpha_1 = np.true_divide(epsilon, 2)
    alhpa_2 = np.true_divide(epsilon, 2) * np.log(np.true_divide(1, delta))

    # z_array = np.array([0.0 for a in np.arange(0, number_of_partition, 1)])

    record_sensitivity = abs(np.true_divide(data, len(data)))

    sample_probability = [0 for a in record_sensitivity]
    for round in np.arange(0, len(record_sensitivity), 1):
        if record_sensitivity[round] > constant:
            sample_probability[round] = np.true_divide((1 - np.exp(epsilon)), (
                    1 - np.exp(epsilon * np.true_divide(record_sensitivity[round], constant))))
        else:
            sample_probability[round] = 1
    # sample_probability = np.true_divide((1 - np.exp(epsilon)),(1 - np.exp(np.true_divide(epsilon * record_sensitivity, constant))))
    # print "sample_probability", sample_probability
    sampled_data = random_pick_probability(data, sample_probability)

    real_result = np.average(sampled_data)
    # print "sampled_data",sampled_data
    local_sensitivity = np.true_divide(np.max(np.abs(sampled_data)), len(sampled_data))
    # print "local_sensitivity",local_sensitivity
    smooth_sensitivity = local_sensitivity * np.exp(beta)
    # print "smooth_sensitivity:", smooth_sensitivity
    noise_result = real_result +  np.random.laplace(0, np.true_divide(local_sensitivity, epsilon))
    return noise_result, real_result


####################################################################
# experiment for private budget
# laplace mechainsm
####################################################################
def laplace_mechaism(data,epsilon = 1.0,delta = 0.00001,number_of_partition = 5,beta = 0.5,constant=2):
    alpha_1 = np.true_divide(epsilon, 2)
    alhpa_2 = np.true_divide(epsilon, 2) * np.log(np.true_divide(1, delta))

    real_result = np.average(data)
    # print "data",data
    local_sensitivity = np.true_divide(np.max(np.abs(data)),len(data))
    # print "local_sensitivity",local_sensitivity
    smooth_sensitivity = local_sensitivity * np.exp(beta)
    # print "smooth_sensitivity:", smooth_sensitivity
    noise_result = real_result + np.true_divide(local_sensitivity, alpha_1) * np.random.laplace(0, 1)
    return noise_result,real_result

####################################################################
# experiment to find out best balance parameter
# this experimetn is not only relatd to balance parameter
# but also related to privacy budget.
####################################################################
def best_balance_parameter(data,number_of_points = 50,name=""):
    # loop = 100
    loop = 100
    record_sensitivity = abs(np.true_divide(data, len(data)))
    # print "sorted record_sensitivity:",np.sort(record_sensitivity)
    c_array = np.linspace(np.average(record_sensitivity),np.max(record_sensitivity),number_of_points)
    # c_array = np.linspace(0.5,2.5, number_of_points)
    # print "c_array",c_array
    epsilon_array = np.linspace(0.01, 0.5, 10)
    # epsilon_array = np.array([np.linspace(0.05,1,10),np.linspace(1,5,10)])
    # print "eppsilon_array:",epsilon_array
    c_array,epsilon_array = np.meshgrid(c_array,epsilon_array)
    # print "c_array:",c_array[0:2,:]
    # print "sum:", np.sum(c_array[0:1,:],axis=0)
    error = np.zeros([len(c_array),len(c_array[0])])

    for index1 in np.arange(0,len(c_array),1):
        for index2 in np.arange(0,len(c_array[0]),1):
            tmp_error = 0
            for count in np.arange(0, loop, 1):
                tmp = personlized_sample(data, epsilon=epsilon_array[index1][index2], constant=c_array[index1][index2])
                tmp_error = tmp_error + np.square(tmp[0] - tmp[1])
            # print "tmp_error:",tmp_error
            error[index1][index2] = np.sqrt(np.true_divide(tmp_error, loop))


    index_min = 0
    average_error = np.sum(error[0:1,:],axis=0)
    for index in np.arange(0, len(average_error), 1):
        if average_error[index] < average_error[index_min]:
            index_min=index
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(epsilon_array, c_array, error, rstride=1, cstride=1, cmap='rainbow')
    # ax.set_xlabel("privacy budget", fontsize=font_size)
    # ax.set_ylabel("balance parameter", fontsize=font_size)
    # ax.set_zlabel("error", fontsize=font_size)
    # # plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,bottom=0.13, top=0.91)
    # # plt.tight_layout()
    # for angle in range(0, 360, 10):
    #     # ax.set_zlabel("Angle: " + str(angle))
    #     ax.view_init(30, angle)
    #     filename = "./" + str(angle) + ".png"
    #     # plt.show()
    #     plt.savefig("./figure/best_balance_parameter_average/" + filename)
    # f = open("./figure/best_balance_parameter_average/best_balance_average.txt","w")
    # f.write("best balance parameter(average function) ="+str(c_array[0][index_min]))
    # f.close()
    return c_array[0][index_min]

####################################################################
# experiment to find out best balance parameter
# this experimetn is not only relatd to balance parameter
# but also related to privacy budget.
####################################################################
def best_balance_parameter_for_aggregation(data,number_of_points = 50,name=""):
    # loop = 100
    loop = 100
    record_sensitivity = abs(np.true_divide(data, len(data)))
    # print "sorted record_sensitivity:",np.sort(record_sensitivity)
    c_array = np.linspace(np.average(record_sensitivity),np.max(record_sensitivity),number_of_points)
    # c_array = np.linspace(0.5,2.5, number_of_points)
    # print "c_array",c_array
    epsilon_array = np.linspace(0.01, 0.5, 10)
    # epsilon_array = np.array([np.linspace(0.05,1,10),np.linspace(1,5,10)])
    # print "eppsilon_array:",epsilon_array
    c_array,epsilon_array = np.meshgrid(c_array,epsilon_array)
    # print "c_array:",c_array[0:2,:]
    # print "sum:", np.sum(c_array[0:1,:],axis=0)
    error = np.zeros([len(c_array),len(c_array[0])])

    for index1 in np.arange(0,len(c_array),1):
        for index2 in np.arange(0,len(c_array[0]),1):
            tmp_error = 0
            for count in np.arange(0, loop, 1):
                tmp = personalized_sample_aggreation(data, epsilon=epsilon_array[index1][index2], constant=c_array[index1][index2])
                tmp_error = tmp_error + np.square(np.abs(tmp[0] - tmp[1]))
            # print "tmp_error:",tmp_error
            error[index1][index2] = np.sqrt(np.true_divide(tmp_error, loop))


    index_min = 0
    average_error = np.sum(error[0:1,:],axis=0)
    for index in np.arange(0, len(average_error), 1):
        if average_error[index] < average_error[index_min]:
            index_min=index
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(epsilon_array, c_array, error, rstride=1, cstride=1, cmap='rainbow')
    # ax.set_xlabel("privacy budget", fontsize=font_size)
    # ax.set_ylabel("balance parameter", fontsize=font_size)
    # ax.set_zlabel("error", fontsize=font_size)
    # # plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,bottom=0.13, top=0.91)
    # # plt.tight_layout()
    # for angle in range(0, 360, 10):
    #     # ax.set_zlabel("Angle: " + str(angle))
    #     ax.view_init(30, angle)
    #     filename = "./" + str(angle) + ".png"
    #     # plt.show()
    #     plt.savefig("./figure/best_balance_parameter_average/" + filename)
    # f = open("./figure/best_balance_parameter_average/best_balance_average.txt","w")
    # f.write("best balance parameter(average function) ="+str(c_array[0][index_min]))
    # f.close()
    return c_array[0][index_min]

####################################################################
####################################################################



def one_data_set(epsilon, mean, standard_deviation, number_of_number, repeate, number_of_thread):
    result_sample_aggreation = 0.0
    result_personlized_sample = 0.0
    result_laplace = 0.0
    result_personalized_sample_aggregation = 0.0

    data = np.random.normal(mean, standard_deviation, number_of_number)
    constant = best_balance_parameter(data, 10)
    constant_aggregation = best_balance_parameter_for_aggregation(data, 10)

    for round in np.arange(0, repeate, 1):
        if round % 10 == 0:
            print "number_of_thread:", number_of_thread, "round:", round
        # print "sample and aggreation:"
        tmp = sample_aggreation(data, epsilon=epsilon)
        result_sample_aggreation += np.square(tmp[0] - tmp[1])
        # print "result of sample_aggreation:", tmp

        # print "personlized sample 1 time:"
        tmp = personalized_sample_aggreation(data, epsilon=epsilon, constant=constant_aggregation)
        result_personalized_sample_aggregation += np.square(np.abs(tmp[0] - tmp[1]))
        # print "personlized_sample_aggreation_1:", tmp

        # print "personlized sample 1 time:"
        tmp = personlized_sample(data, epsilon=epsilon, constant=constant)
        result_personlized_sample += np.square(tmp[0] - tmp[1])
        # print "personlized_sample_aggreation_1:", tmp

        # print "laplace mechanism:"
        tmp = laplace_mechaism(data, epsilon=epsilon)
        result_laplace += np.square(tmp[0] - tmp[1])

    result_sample_aggreation = np.sqrt(np.true_divide(result_sample_aggreation, repeate))
    result_personlized_sample = np.sqrt(np.true_divide(result_personlized_sample, repeate))
    result_laplace = np.sqrt(np.true_divide(result_laplace, repeate))
    result_personalized_sample_aggregation = np.sqrt(np.true_divide(result_personalized_sample_aggregation, repeate))
    return [result_sample_aggreation, result_personlized_sample, result_laplace, result_personalized_sample_aggregation]



if __name__ == '__main__':
    # multiprocessing.freeze_support()
    mean = 10
    standard_deviation = 10
    number_of_number = 100

    privacy_array = np.linspace(0.01, 0.5, 10)
    result_sample_aggreation = np.array([0.0 for a in np.arange(0, len(privacy_array), 1)])
    result_personlized_sample = np.array([0.0 for a in np.arange(0, len(privacy_array), 1)])
    result_laplace = np.array([0.0 for a in np.arange(0, len(privacy_array), 1)])
    result_personalized_sample_aggregation = np.array([0.0 for a in np.arange(0, len(privacy_array), 1)])

    number_of_thread = 20
    number_round = 25
    number_repeate = number_round * number_of_thread
    number_repeate_2 = 500
    # number_of_thread = 5
    # number_round = 1
    # number_repeate = number_round*number_of_thread
    # number_repeate_2 = 50

    for index_1 in np.arange(0, len(privacy_array), 1):
        print index_1, "of", len(privacy_array)
        for round in np.arange(0, number_round, 1):

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            threads = []
            for index_2 in np.arange(0, number_of_thread, 1):
                tmp = pool.apply_async(one_data_set, args=(privacy_array[index_1], mean,
                                                           standard_deviation, number_of_number,
                                                           number_repeate_2, index_2,))
                threads.append(tmp)

            print "waitng........."
            pool.close()
            pool.join()
            # print "join is over:"
            # print "threads:",threads
            for thread in threads:
                # print "t:",t
                tmp = thread.get()
                # print tmp
                result_sample_aggreation[index_1] += tmp[0]
                result_personlized_sample[index_1] += tmp[1]
                result_laplace[index_1] += tmp[2]
                result_personalized_sample_aggregation[index_1] += tmp[3]

        print "result_sample_aggreation:", result_sample_aggreation[index_1]
        print "result_personlized_sample:", result_personlized_sample[index_1]
        print "result_laplace", result_laplace[index_1]
        print "result_personalized_sample_aggregation:", result_personalized_sample_aggregation[index_1]
        result_sample_aggreation[index_1] = (np.true_divide(result_sample_aggreation[index_1], number_repeate))
        result_personlized_sample[index_1] = (np.true_divide(result_personlized_sample[index_1], number_repeate))
        result_laplace[index_1] = (np.true_divide(result_laplace[index_1], number_repeate))
        result_personalized_sample_aggregation[index_1] = (np.true_divide(result_personalized_sample_aggregation[index_1], number_repeate))
        print "result_sample_aggreation:", result_sample_aggreation[index_1]
        print "result_personlized_sample:", result_personlized_sample[index_1]
        print "result_laplace", result_laplace[index_1]
        print "result_personalized_sample_aggregation:", result_personalized_sample_aggregation[index_1]
    print "result_sample_aggreation:", result_sample_aggreation
    print "result_personlized_sample:", result_personlized_sample
    print "result_laplace", result_laplace
    print "result_personalized_sample_aggregation:", result_personalized_sample_aggregation

    fig = plt.figure()
    plt.plot(privacy_array, result_sample_aggreation, 'r-*', markersize=8, linewidth=1, label='sample and aggreation')
    plt.plot(privacy_array, result_personalized_sample_aggregation, 'darkorange',marker="v",markersize=8, linewidth=1, label='personalized sample \n and aggregation')
    plt.plot(privacy_array, result_laplace, 'g-s', markersize=8, linewidth=1, label='Laplace')
    plt.plot(privacy_array, result_personlized_sample, 'b-o', markersize=8, linewidth=1, label='personalized sample \n Laplace')
    plt.legend(loc='upper right', fontsize=font_size)
    # plt.title('comparison of error for average function', fontsize=font_size)
    plt.xlabel('privacy budget', fontsize=font_size)
    plt.xticks(np.array([0.1,0.2,0.3,0.4,0.5]),fontsize=font_size )
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel('RMSE', fontsize=font_size)
    plt.tight_layout()
    plt.savefig('../figure/thread_average.png',dpi=600, bbox_extra_artists=(),bbox_inches='tight')
    plt.show()




