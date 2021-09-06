#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
import timeit as timeit
import  time
from mpl_toolkits.mplot3d import Axes3D




def random_pick_probability(numbers,probabilities):
    prob =  np.random.uniform(0,1,len(numbers))
    select = np.array(probabilities) - prob
    # print "select:",select>0
    return np.array(numbers)[select>0]

# np.random.seed(100)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
mean =10
standard_deviation =100
number_of_number = 100
font_size = 14


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
def best_balance_parameter(data,number_of_points = 50):
    loop = 200
    record_sensitivity = abs(np.true_divide(data, len(data)))
    # print "sorted record_sensitivity:",np.sort(record_sensitivity)
    c_array = np.linspace(np.min(record_sensitivity),np.max(record_sensitivity),number_of_points)
    # c_array = np.linspace(0.5,2.5, number_of_points)
    # print "c_array",c_array
    # epsilon_array = np.hstack((np.linspace(0.05,1,10),np.linspace(1,5,10)))
    epsilon_array = np.linspace(0.1, 0.5, 10)
    # epsilon_array = np.array([np.linspace(0.05,1,10),np.linspace(1,5,10)])
    # print "eppsilon_array:",epsilon_array
    c_array,epsilon_array = np.meshgrid(c_array,epsilon_array)
    # print "c_array:",c_array[0:2,:]
    # print "sum:", np.sum(c_array[0:1,:],axis=0)
    error = np.zeros([len(c_array),len(c_array[0])])

    for index1 in np.arange(0,len(c_array),1):
        print "index1:", index1, "len(c_array):", len(c_array)
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
    path = "./figure/best_balance_parameter_average/"
    if not os.path.exists(path):
        os.mkdir(path)
    np.savetxt(path + "real_average_epsilon.txt" , epsilon_array)
    np.savetxt(path + "real_average_c_array.txt"   , c_array)
    np.savetxt(path + "real_average_error.txt"  , error)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(epsilon_array, c_array, error, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel("privacy budget", fontsize=font_size)
    ax.set_ylabel("balance parameter", fontsize=font_size)
    ax.set_zlabel("error", fontsize=font_size)
    # plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,bottom=0.13, top=0.91)
    # plt.tight_layout()
    for angle in range(0, 360, 10):
        # ax.set_zlabel("Angle: " + str(angle))
        ax.view_init(30, angle)
        filename = "./" + str(angle) + ".png"
        # plt.show()
        plt.savefig("./figure/best_balance_parameter_average/" + filename)
    f = open("./figure/best_balance_parameter_average/best_balance_average.txt","w")
    f.write("best balance parameter(average function) ="+str(c_array[0][index_min]))
    f.close()
    return c_array[0][index_min]

####################################################################
# experiment to find out best balance parameter
# this experimetn is not only relatd to balance parameter
# but also related to privacy budget.
####################################################################
def best_balance_parameter(data,number_of_points = 50):
    loop = 200
    record_sensitivity = abs(np.true_divide(data, len(data)))
    # print "sorted record_sensitivity:",np.sort(record_sensitivity)
    c_array = np.linspace(np.min(record_sensitivity),np.max(record_sensitivity),number_of_points)
    # c_array = np.linspace(0.5,2.5, number_of_points)
    # print "c_array",c_array
    # epsilon_array = np.hstack((np.linspace(0.05,1,10),np.linspace(1,5,10)))
    epsilon_array = np.linspace(0.1, 0.5, 10)
    # epsilon_array = np.array([np.linspace(0.05,1,10),np.linspace(1,5,10)])
    # print "eppsilon_array:",epsilon_array
    c_array,epsilon_array = np.meshgrid(c_array,epsilon_array)
    # print "c_array:",c_array[0:2,:]
    # print "sum:", np.sum(c_array[0:1,:],axis=0)
    error = np.zeros([len(c_array),len(c_array[0])])

    for index1 in np.arange(0,len(c_array),1):
        print "index1:", index1, "len(c_array):", len(c_array)
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
    path = "./figure/best_balance_parameter_average/"
    if not os.path.exists(path):
        os.mkdir(path)
    np.savetxt(path + "real_average_epsilon.txt" , epsilon_array)
    np.savetxt(path + "real_average_c_array.txt"   , c_array)
    np.savetxt(path + "real_average_error.txt"  , error)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(epsilon_array, c_array, error, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel("privacy budget", fontsize=font_size)
    ax.set_ylabel("balance parameter", fontsize=font_size)
    ax.set_zlabel("error", fontsize=font_size)
    # plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,bottom=0.13, top=0.91)
    # plt.tight_layout()
    for angle in range(0, 360, 10):
        # ax.set_zlabel("Angle: " + str(angle))
        ax.view_init(30, angle)
        filename = "./" + str(angle) + ".png"
        # plt.show()
        plt.savefig("./figure/best_balance_parameter_average/" + filename)
    f = open("./figure/best_balance_parameter_average/best_balance_average.txt","w")
    f.write("best balance parameter(average function) ="+str(c_array[0][index_min]))
    f.close()
    return c_array[0][index_min]



####################################################################
####################################################################
mean =10
standard_deviation =100
number_of_number = 100
data = np.random.normal(mean, standard_deviation, number_of_number)
constant = best_balance_parameter(data,20)
print "constant:",constant

path = "./figure/best_balance_parameter_average/"
epsilon_array = np.loadtxt(path + "real_average_epsilon.txt"  )
c_array = np.loadtxt(path + "real_average_c_array.txt"  )
error = np.loadtxt(path + "real_average_error.txt" )


# privacy_array = np.linspace(0.01,0.5,20)
privacy_array = np.linspace(0.1, 0.5, 10)
result_sample_aggreation = np.array([0.0 for a in np.arange(0,len(privacy_array) ,1)])
result_personlized_sample = np.array([0.0 for a in np.arange(0,len(privacy_array) ,1)])
result_laplace = np.array([0.0 for a in np.arange(0,len(privacy_array) ,1)])
result_personalized_sample_aggregation = np.array([0.0 for a in np.arange(0,len(privacy_array) ,1)])

number_repeate = 1000
for index_1 in np.arange(0,len(privacy_array),1):
    print "privacy:", privacy_array[index_1], "len(privacy_array)", len(privacy_array)
    index_c = np.argmin(error[index_1])
    constant = c_array[index_1][index_c]
    for index_2 in np.arange(0,number_repeate,1):
        #print "sample and aggreation:"
        tmp = sample_aggreation(data,epsilon=privacy_array[index_1])
        result_sample_aggreation[index_1] += np.square(tmp[0] - tmp[1])
        #print "result of sample_aggreation:", tmp

        # # print "personlized sample 1 time:"
        # tmp = personalized_sample_aggreation(data, epsilon=privacy_array[index_1], constant=constant)
        # result_personalized_sample_aggregation[index_1] += np.square(tmp[0] - tmp[1])
        # # print "personlized_sample_aggreation_1:", tmp

        #print "personlized sample 1 time:"
        tmp = personlized_sample(data,epsilon=privacy_array[index_1],constant=constant)
        result_personlized_sample[index_1] += np.square(tmp[0] - tmp[1])
        #print "personlized_sample_aggreation_1:", tmp

        #print "laplace mechanism:"
        tmp = laplace_mechaism(data,epsilon=privacy_array[index_1])
        result_laplace[index_1] += np.square(tmp[0] - tmp[1])
        # print "result of personlized:", tmp
    result_sample_aggreation[index_1] = np.sqrt(np.true_divide(result_sample_aggreation[index_1],number_repeate))
    result_personlized_sample[index_1] = np.sqrt(np.true_divide(result_personlized_sample[index_1], number_repeate))
    result_laplace[index_1] = np.sqrt(np.true_divide(result_laplace[index_1], number_repeate))
    result_personalized_sample_aggregation[index_1] = np.sqrt(np.true_divide(result_personalized_sample_aggregation[index_1], number_repeate))

fig = plt.figure()
plt.plot(privacy_array, result_sample_aggreation, 'r-', linewidth=2, label='sample and aggreation')
plt.plot(privacy_array, result_personalized_sample_aggregation, 'darkorange', linewidth=2, label='personalized sample \n and aggregation')
plt.plot(privacy_array, result_laplace, 'g-', linewidth=2, label='Laplace')
plt.plot(privacy_array, result_personlized_sample, 'b-', linewidth=2, label='personalized sample \n Laplace')
plt.legend(loc='upper right',fontsize=font_size)
# plt.title('comparison of error for average function')
plt.xlabel('privacy budget',fontsize=font_size)
plt.xticks(np.array([0.1,0.2,0.3,0.4,0.5]),fontsize=font_size )
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel('RMSE',fontsize=font_size)
plt.tight_layout()
plt.savefig('./figure/average.png')
plt.show()







# data = np.random.normal(mean, standard_deviation, number_of_number)
# # print "laplace_mechanism:",laplace_mechaism(data)

# number_repeate = 10000
# result_sample_aggreation = np.array([0 for a in np.arange(0,number_repeate ,1)])
# # result_personlized_sample_aggreation = np.array([0 for a in np.arange(0,number_repeate ,1)])
# result_personlized_sample = np.array([0 for a in np.arange(0,number_repeate ,1)])
# result_laplace = np.array([0 for a in np.arange(0,number_repeate ,1)])
# for loop in np.arange(0,number_repeate ,1):
#     data = np.random.normal(mean, standard_deviation, number_of_number)
#     print "sample and aggreation:"
#     tmp = sample_aggreation(data)
#     result_sample_aggreation [loop] = tmp[0]-tmp[1]
#     print "result of sample_aggreation:", tmp
# 
#     # print "personlized sample and aggreation:"
#     # tmp = personlized_sample_aggreation(data)
#     # result_personlized_sample_aggreation[loop] = tmp[0]-tmp[1]
#     # print "result of personlized:", tmp
# 
#     print "personlized sample 1 time:"
#     tmp = personlized_sample(data)
#     result_personlized_sample[loop] = tmp[0] - tmp[1]
#     print "personlized_sample_aggreation_1:", tmp
# 
#     print "laplace mechanism:"
#     tmp = laplace_mechaism(data)
#     result_laplace[loop] = tmp[0] - tmp[1]
#     print "result of personlized:", tmp
# 
# print "sample_aggreation:",result_sample_aggreation ,"\n","result_personlized_sample_aggreation:",result_personlized_sample_aggreation,"\n"
# print "result_personlized_sample:",result_personlized_sample,"\n","result_laplace:",result_laplace
# 
# print "sample_aggreation:",np.sum(np.abs(result_sample_aggreation )),"\n","result_personlized_sample_aggreation:",np.sum(np.abs(result_personlized_sample_aggreation)),"\n"
# print "result_personlized_sample:",np.sum(np.abs(result_personlized_sample)),"\n","result_laplace:",np.sum(np.abs(result_laplace))


#
# epsilon_array = np.linspace(0.05,1.0,20)
# print epsilon_array[0]
# average_laplace_rmse_array = np.array([0.0 for a in epsilon_array])
# average_sdp_rmse_array = np.array([0.0 for a in epsilon_array])
# mean_real_array = np.array([0.0 for a in epsilon_array])
# loop = 500
# constant = 2








####################################################################
# experiment for private budget
####################################################################
# epsilon_array = np.linspace(0.05,1.0,20)
# print epsilon_array[0]
# average_laplace_rmse_array = np.array([0.0 for a in epsilon_array])
# average_sdp_rmse_array = np.array([0.0 for a in epsilon_array])
# mean_real_array = np.array([0.0 for a in epsilon_array])
# loop = 500
# constant = 2
#
# for index in np.arange(0,len(epsilon_array),1):
#     for round in np.arange(0, loop, 1):
#         data = np.random.normal(mean, standard_deviation, number_of_number)
#         mean_real_array[index] = np.average(data)
#         print "mean_real:", mean_real_array[index]
#
#         local_sensitivity = np.max(abs(data)) / len(data)
#         print "local_sensitivity:", local_sensitivity
#
#         average_laplace_rmse_array[index] += np.square( np.random.laplace(0, local_sensitivity / epsilon_array[index]) )
#         print "average_laplace:",  average_laplace_rmse_array[index]
#
#         record_sensitivity = abs(np.true_divide(data, len(data)))
#         # print "record_sensitivity:",record_sensitivity
#         sample_probability = [0 for a in record_sensitivity]
#         for round in np.arange(0, len(record_sensitivity), 1):
#             if record_sensitivity[round]>constant:
#                 sample_probability[round] = np.true_divide((1 - np.exp(epsilon_array[index])), (1 - np.exp(epsilon_array[index] * np.true_divide(record_sensitivity[round],constant))))
#             else:
#                 sample_probability[round] = 1
#         sample_probability = np.true_divide((1 - np.exp(epsilon_array[index])), (1 - np.exp(np.true_divide(epsilon_array[index] * record_sensitivity,constant))))
#         # print "sample_probability", sample_probability
#         sampled_data = random_pick_probability(data, sample_probability)
#         average_sdp_rmse_array[index] += np.square(np.average(sampled_data) + np.random.laplace(0, constant / epsilon_array[index]) - mean_real_array[index])
#         print "average_sdp:", average_sdp_rmse_array[index]
#     average_laplace_rmse_array[index] = np.sqrt(np.true_divide(average_laplace_rmse_array[index], loop))
#     average_sdp_rmse_array[index] = np.sqrt(np.true_divide(average_sdp_rmse_array[index],  loop))
#
# # print  "mean_real_array:",mean_real_array/loop
# print  "average_laplace_rmse_array:",average_laplace_rmse_array
# print  "average_sdp_rmse_array:",average_sdp_rmse_array
#
# # figure about effect of epsilon on accuracy。
# # plt.subplot(1, 2, 1)
# plt.plot(epsilon_array, average_laplace_rmse_array, 'r-', linewidth=2, label='Laplace mechanism')
# plt.plot(epsilon_array, average_sdp_rmse_array, 'b-', linewidth=2, label='SDP mechanism')
# plt.legend(loc='upper right',fontsize=font_size)
# # plt.title('Influence of private budget \n on rmse ')
# plt.xlabel('privacy budget',fontsize=font_size)
# plt.xlim(0)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.ylabel('RMSE',fontsize=font_size)
# plt.tight_layout()
# plt.savefig('./figure/isdp_privacy_budget_rmse.png')
# plt.show()






