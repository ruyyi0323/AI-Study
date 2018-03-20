def generate_model():
    '''
    Define the graph, cpt, and the variable states
    :return: the graph, the conditional probability table, and the variable states
    '''
    graph = {
        "amenities": ["location"],
        "neighborhood": ["location", "children"],
        "children": ["schools"],
        "location": ["age", "price"],
        "age": ["price"],
        "schools": ["price"],
        "size": ["price"],
        "price": []
    }
    cpt = {
        "amenities": [0.3, 0.7],
        "neighborhood": [0.4, 0.6],
        "location": [[0.3, 0.4, 0.3], [0.8, 0.15, 0.05], [0.2, 0.4, 0.4], [0.5, 0.35, 0.15]],
        "children": [[0.6, 0.4], [0.3, 0.7]],
        "schools": [[0.7, 0.3], [0.8, 0.2]],
        "age": [[0.3, 0.7], [0.6, 0.4], [0.9, 0.1]],
        "price": [[0.5, 0.4, 0.1], [0.4, 0.45, 0.15], [0.35, 0.45, 0.2],
                  [0.4, 0.3, 0.3], [0.35, 0.3, 0.35], [0.3, 0.25, 0.45],
                  [0.45, 0.4, 0.15], [0.4, 0.45, 0.15], [0.35, 0.45, 0.2],
                  [0.25, 0.3, 0.45], [0.2, 0.25, 0.55], [0.1, 0.2, 0.7],
                  [0.7, 0.299, 0.001], [0.65, 0.33, 0.02], [0.65, 0.32, 0.03],
                  [0.55, 0.35, 0.1], [0.5, 0.35, 0.15], [0.45, 0.4, 0.15],
                  [0.6, 0.35, 0.05], [0.55, 0.35, 0.1], [0.5, 0.4, 0.1],
                  [0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
                  [0.8, 0.1999, 0.0001], [0.75, 0.24, 0.01], [0.75, 0.23, 0.02],
                  [0.65, 0.3, 0.05], [0.6, 0.33, 0.07], [0.55, 0.37, 0.08],
                  [0.7, 0.27, 0.03], [0.64, 0.3, 0.06], [0.61, 0.32, 0.07],
                  [0.48, 0.42, 0.1], [0.41, 0.39, 0.2], [0.37, 0.33, 0.3]],
        "size": [0.33, 0.34, 0.33],
    }

    var_card = {
        "amenities": ["lots", "little"],
        "neighborhood": ["bad", "good"],
        "location": ["good", "bad", "ugly"],
        "children": ["bad", "good"],
        "size": ["small", "medium", "large"],
        "schools": ["bad", "good"],
        "age": ["old", "new"],
        "price": ["cheap", "ok", "expensive"]
    }
    return graph, cpt, var_card


def get_command():
    '''
    analysis the input command
    :return: the node to query, the evidence node and its state, number of samples to generate, number of discard samples
    '''
    evidence_node = []
    parse = argparse.ArgumentParser()
    parse.add_argument("sampling_method", action="store", type=str)
    parse.add_argument("node_to_query", action="store", type=str)
    parse.add_argument("evidence_set", nargs='*', type=str)
    parse.add_argument('-u', option_strings='-u', type=int, help='Number of Updates to be made')
    parse.add_argument('-d', option_strings='-d', type=int,
                       help='Number of Updates to ignore before computing probability')

    args = parse.parse_args()
    try:
        node_to_query = args.node_to_query
    except:
        print('no nodes to query')
    try:
        evidence_set = args.evidence_set
    except:
        print('no given evidence')
    try:
        num_update = args.u
    except:
        print('no drop number')
    try:
        num_drop = args.d
    except:
        print('no nodes to query')

    for item in evidence_set:
        item = item.split("=")
        evidence_node.append(item)
    evidence_node = dict(evidence_node)

    return node_to_query, evidence_node, num_update, num_drop


def setup_samples():
    '''
    setup initial sample and its state
    :return: the initial sample and its state
    '''
    sample = {}
    for key in graph:
        sample.update({key: var_card[key][random.randint(0, len(var_card[key]) - 1)]})
    sample.update(evidence_set)
    return sample


def get_parents(node):
    '''
    return the parent node of a specific node
    :param node: input node to query
    :return: a list of parent nodes
    '''
    parent_list = []
    for key, value in graph.items():
        if node in value: parent_list.append(key)
    return parent_list


def get_children(node):
    '''
    get the children node of the specific node
    :param node: input a specific node to get children
    :return: a list of children nodes
    '''
    for keys, value in graph.items():
        if node in keys: return value


def get_updatable_nodes():
    '''
    get a list of nodes that available for update 
    (available update node should innclude every node in the graph, except for the evidence node)
    :return: 
    '''
    return list(set(list(graph.keys())) ^ set(list(evidence_set.keys())))


def pick_nodes(node_list):
    '''
    pick a random node from the updatable node list
    :param node_list: input the updataable node list
    :return: a random node
    '''
    return node_list[random.randint(0, len(node_list) - 1)]


def get_status_probability(sample, node, condition):
    '''
    calculate the prior probability condition on other nodes in the markov blanket (PPT24)
    P(new_state) = P(condition = prior | sample(old_state)) * P(condition = post | sample(old_state))
    :param sample: input the sample that used for calculation
    :param node: input the node that is going to calculate the new state
    :param condition: the mode of calculation, if it is prior, calculation based on the parents' state, if it is post
    calculation based on the children and parents of children's state
    :return: the production of P(condition = prior | sample(old_state)) * P(condition = post | sample(old_state))
    '''
    if condition == 'prior':
        condition_list = list(sample.keys())
        condition_list.remove(node)
        condition_dict = build_condition_dict(condition_list, sample)
        probability = get_cpt_row(node, condition_dict, 'all')
        return np.array(probability)
    if condition == 'post':
        probability = list(np.ones(len(var_card[node])))
        children_list = get_children(node)
        for children in children_list:
            col_index = var_card[children].index(sample[children])
            condition_list = list(sample.keys())
            condition_list.remove(children)
            condition_dict = build_condition_dict(condition_list, sample)
            for col in range(len(var_card[node])):
                condition_dict[node] = col
                probability[col] = probability[col] * get_cpt_row(children, condition_dict, col_index)
        return np.array(probability)


def build_condition_dict(condition_list, sample):
    '''
    change the state from string to integer, remove the node under update, make further calculation more convenience
    :param condition_list: input the nodes' state in sample
    :param sample: input the sample
    :return: a dictionary including the nodes and its state
    '''
    condition_dict = {}
    for node_iter in condition_list:
        condition_dict.update({node_iter: var_card[node_iter].index(sample[node_iter])})
    return condition_dict


def get_cpt_row(node, status_dict, col_index):
    '''
    fetch the conditional probability in the cpt to make productions
    :param node: the node that required to get the probability from cpt
    :param status_dict: all the conditional states from other nodes
    :param col_index: if the node doesn't have any parent node, use its own prior probability from the cpt,
    since we retrieve value of probability from the cpt as a list, we set 'all' to nodes in this case.
    :return: a list of probability (correspond to the different states of the node)
    '''
    row_index = 0
    if node == 'amenities': return cpt[node]
    if node == 'neighborhood': return cpt[node]
    if node == 'size': return cpt[node]
    if node == 'schools': row_index = status_dict['children']
    if node == 'location': row_index = status_dict['amenities'] * 2 \
                                       + status_dict['neighborhood']
    if node == 'children': row_index = status_dict['neighborhood']
    if node == 'age': row_index = status_dict['location']
    if node == 'price': row_index = status_dict['location'] * 12 \
                                    + status_dict['age'] * 6 \
                                    + status_dict['schools'] * 3 \
                                    + status_dict['size']
    if col_index == 'all':
        return cpt[node][row_index]
    else:
        return cpt[node][row_index][col_index]


def normalization(array):
    '''
    normalizaiton of the probability (PPT25)
    :param array: normalization the list ofprobability
    :return: a normalized prior probability list
    '''
    return array / max(np.cumsum(array))


def update_procedure(node, sample):
    '''
    update the state of the specific node generated in previous function
    :param node: the node under update
    :param sample: the initiate or latest generated sample (we made calculation based on the state inn this sample)
    :return: 
    '''
    # parent = get_parents(node)
    # children = get_children(node)

    prior_prob = get_status_probability(sample, node, 'prior')
    if len(get_children(node)) != 0:
        post_prob = get_status_probability(sample, node, 'post')
    else:
        post_prob = 1.0
    prob = normalization(post_prob * prior_prob) #ppt24 and ppt25 calculation of the prior probability
    assignment_dist = np.cumsum(prob) - random.random()
    assignment_status = var_card[node][len(np.where(assignment_dist <= 0)[0])]
    #assign the new state based on the probability generated before
    new_sample = dict(sample)
    new_sample[node] = assignment_status #update the state to the sample
    return new_sample #return new sample
    # sample[node] = assignment_status is wrong


def prob_calculation(samples, node, iteration, discard):
    '''
    calculation of the probability based on all the sample we have (already removed the discarded sample)
    :param samples: a list of sample we have
    :param node: node to query
    :param iteration: the number of updates, AKA iteration
    :param discard: the number of samples we removed from sample list.
    :return: the probability of the node to query, correspond to its state
    '''
    probability = {}
    for values in var_card[node]:
        probability.update({values: 0})
    for sample in samples:
        probability[sample[node]] += 1
    for key, value in probability.items():
        probability[key] = probability[key] / (iteration - discard)
    return probability


def gibbs_sampling(num_update, num_drop, fig_data, ways):
    '''
    main procedure of the gibbs sampling
    :param num_update: number of update (the maximum iteration record)
    :param num_drop: 
    :param fig_data: trace the change of the probability with different iteration and discard number, 
    record them into the fig data
    :param ways: not using this variable
    :return: 
    '''
    sample_list = []

    # parent = get_parents('amenities')
    # children = get_children('amenities')
    # parents_of_children = get_parents_of_children('amenities')

    curr_sample = setup_samples()#generate initial sample, randomly assign state to them
    # sample_list.append(curr_sample)
    updatable_nodes = get_updatable_nodes()# get updatable nodes
    if ways == 'update':
        for iteration in range(1, num_update+1):
            update_node = pick_nodes(updatable_nodes)
            curr_sample = update_procedure(update_node, curr_sample)# update the state and get the new sample
            if iteration > num_drop: #avoid the initial samples generated, the quantity is controled by num_drop
                sample_list.append(curr_sample)
                if iteration % 1000 == 0:
                    probability = prob_calculation(sample_list, node_to_query, iteration, num_drop)
                    # calculate the final probability
                    fig_data_process(fig_data, probability, iteration) #record all the samples
                    # print("Iteration = %d, sample drop = %d" % (iteration, num_drop))
        format_printing(node_to_query, iteration, num_drop, probability)# printing
    # if ways == 'discard':
    #     for iteration in range(0, num_update+1):
    #         update_node = pick_nodes(updatable_nodes)
    #         curr_sample = update_procedure(update_node, curr_sample)
    #         sample_list.append(curr_sample)
    #     for discard in range(0, num_drop+1, 100):
    #         sample_list = sample_list[int(0.01*num_drop):]
    #         probability = prob_calculation(sample_list, node_to_query, num_update, discard)
    #         fig_data_process(fig_data, probability, discard)
    #         # print("Discard = %d, sample number = %d" % (discard, num_update))
    #     format_printing(node_to_query, num_update, discard, probability)
    fig_data = np.array(fig_data)
    return fig_data


def format_printing(node, iteration, discard, probability):
    '''
    print the probability with the format
    :param node: 
    :param iteration: 
    :param discard: 
    :param probability: 
    :return: 
    '''
    print('With %d samples and %d discarded, the gibbs sampling gives:' % (iteration, discard))
    for value in probability:
        print('P(%s = %s) = %.4f ' % (node, value, round(probability[value], 4)))
    print('\n')


def fig_data_process(data, probability, iteration):
    '''
    record the result of the calculation of the probability, with different iteration time
    :param data: 
    :param probability: 
    :param iteration: 
    :return: 
    '''
    # temp = list(probability.values())
    data.append(list(probability.values()) + [iteration])


def draw_figure(x_vector, data, xlabel, ylabel):
    '''
    draw all the probability change of all the labels in this variable
    :param x_vector: 
    :param data: 
    :param xlabel: 
    :param ylabel: 
    :return: 
    '''
    for col in range(data.shape[1]):
        plt.plot(x_vector, data[:, col], label=var_card[node_to_query][col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.grid()
    plt.ylim(0,1)

def draw_figure_one(x_vector, data, label, ylabel, col):
    '''
    draw only one label probability change in this variable
    :param x_vector: 
    :param data: 
    :param label: 
    :param ylabel: 
    :param col: 
    :return: 
    '''

    ax1 = plt.subplot(211)
    plt.plot(x_vector, data[:, 0], label=label, c = col, alpha = 0.7)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    ax2 = plt.subplot(212)
    plt.semilogx(x_vector, data[:, 0], label=label, c = col, alpha = 0.7)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

if __name__ == '__main__':
    import random
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    graph, cpt, var_card = generate_model()
    node_to_query, evidence_set, num_update, num_drop = get_command()

    # Example of command:
    # python gibbs_sampling.py gibbs price schools=good location=ugly -u 100000 -d 0
    # python gibbs_sampling.py gibbs amenities location=bad neighbborhood=good -u 100000 -d 1000

    print('---------Analysing Incoming Command---------')
    fig_data =[]
    fig_data = gibbs_sampling(num_update, num_drop, fig_data, 'update')

    print('---------Converge Test---------')

    plt.figure()
    plt.title('Converge Speed Test, Y axis scale is not [0,1]!!!')
    num_update = 100000
    num_drop_list = [0,100,1000,10000]
    repeat_time = 3
    for num_drop in num_drop_list:
        col = 'C'+str(2*num_drop_list.index(num_drop))
        for iter in range(repeat_time):
            fig_data = []
            fig_data = gibbs_sampling(num_update, num_drop, fig_data, 'update')
            draw_figure_one(fig_data[:, len(var_card[node_to_query])]-num_drop, fig_data[:, 0:len(var_card[node_to_query])],
                    'drop = %d'%num_drop, 'Estimate Probability', col)
    # Customize the major grid
    plt.show()

    # plt.subplot(242)
    # num_update, num_drop = 100000, 1000
    # fig2_data = gibbs_sampling(num_update, num_drop, fig2_data, 'update')
    # draw_figure(fig2_data[:, len(var_card[node_to_query])]-num_drop, fig2_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    # plt.subplot(243)
    # num_update, num_drop = 100000, 5000
    # fig3_data = gibbs_sampling(num_update, num_drop, fig3_data, 'update')
    # draw_figure(fig3_data[:, len(var_card[node_to_query])]-num_drop, fig3_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    # plt.subplot(244)
    # num_update, num_drop = 100000, 10000
    # fig4_data = gibbs_sampling(num_update, num_drop, fig4_data, 'update')
    # draw_figure(fig4_data[:, len(var_card[node_to_query])]-num_drop, fig4_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    #
    # fig1_data, fig2_data, fig3_data, fig4_data = [], [], [], []
    # plt.subplot(245)
    # num_update, num_drop = 10000, 1000
    # fig1_data = gibbs_sampling(num_update, num_drop, fig1_data, 'update')
    # draw_figure(fig1_data[:, len(var_card[node_to_query])]-num_drop, fig1_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    # plt.subplot(246)
    # num_update, num_drop = 20000, 1000
    # fig2_data = gibbs_sampling(num_update, num_drop, fig2_data, 'update')
    # draw_figure(fig2_data[:, len(var_card[node_to_query])]-num_drop, fig2_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    # plt.subplot(247)
    # num_update, num_drop = 50000, 1000
    # fig3_data = gibbs_sampling(num_update, num_drop, fig3_data, 'update')
    # draw_figure(fig3_data[:, len(var_card[node_to_query])]-num_drop, fig3_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')
    # plt.subplot(248)
    # num_update, num_drop = 100000, 1000
    # fig4_data = gibbs_sampling(num_update, num_drop, fig4_data, 'update')
    # draw_figure(fig4_data[:, len(var_card[node_to_query])]-num_drop, fig4_data[:, 0:len(var_card[node_to_query])],
    #             'Iteration(drop = %d)'%num_drop, 'Estimate probability')


