
# simple imports
# --------------
import copy
import sys
import argparse


# ### 1. Given a set of friends who have pending financial transactions between them, write an algorithm that minimizes the total cash flow among all the friends.


# main function to find borrowing relationships with least cash flow
# custom algo
# ------------------------------------------------------------------

def cash_flow(r_in):
    
    '''
    
    input in format - relation is a list of tuple in format (giver, taker, value)
    
    example:
    r = [['p0','p1',1], ['p0','p2',2], ['p1','p2',5], ['p1','p3',1]]
    
    Algorithm - custom algo based on iterative edge reduction. 

    Basic idea - 

    1. Initially, find net value people of people. We will call this original net list.
    
    2. In NO particular order, we will iter through all persons - 

    - for each direct outpayee, find all their first order outpayees. Example if p1 > p2,p3 and p2>p4 and p3>p5, 
    then p1's second order outpayment persons are p4,p5. (Read > symbol as "owes")
    
    - pay off p4,p5 and adjust the connections between p1&p2, p2&p4 and p3&p5. Depending on value paid, 
    some of these connections may become void. 
    
    - adjust relation between p1,p2,p3,p4 & p5 accordingly.

    3. Step onto next person in the original net list and carry out the above steps.

    4. Do these iterative steps until each of the persons do NOT have any second order outpayees.

    5. After completion of loop, take a net of all people based on current relationships and make sure it is the same as original net list.

    
    '''
    
    # 0. inits
    # --------
    r = copy.deepcopy(r_in)
    #print(type(r))
    #print(r)
    steps = 0
    total_cash_flow_orig = 0
    
    # 1. net values of all
    # orig_net_list
    # iter and populate simple dict
    # -----------------------------
    net_dict = {}

    for each in r:

        # 1. add incoming
        # ----------------
        try:
            net_dict[each[1]] += each[2]
        except:
            net_dict[each[1]] = each[2]

        # 2. removing from payer
        # ----------------------
        try:
            net_dict[each[0]] -= each[2]
        except:
            net_dict[each[0]] = -1 * each[2]
        
        # 3. total value
        # --------------
        total_cash_flow_orig += each[2]

    
    '''
    net_dict of form like: 
    
    {'p1': -5, 'p0': -3, 'p2': -1, 'p3': 4, 'p4': 4, 'p5': 1}
    
    '''
    
    # 2. begin of main while loop
    # ---------------------------
    while True:
        
        # 0. inits local to while
        # -----------------------
        change_flag = 0
        steps += 1
        
        # 1. iter through each person in the net_dict
        # parsing per person
        # -------------------------------------------
        for curr_p in net_dict:
            
            ##
            #print('at person ' + curr_p)
            
            
            # 0. local inits inside loop per person
            # -------------------------------------
            first_order_r = []
            sec_order_r = []
            sec_ord_payees = []


            # code to retrive second order payees of curr_p and adjust relationships
            ########################################################################

            # 1. getting first order relationships
            # ------------------------------------
            for each in r:
                if curr_p == each[0]:
                    first_order_r += [each]
                    sec_ord_payees += [each[1]]

            # keeping only unique values
            # --------------------------
            sec_ord_payees = list(set(list(sec_ord_payees)))

            # 2. getting secnid order relationships
            # -------------------------------------
            for each in r:
                if each[0] in sec_ord_payees:
                    sec_order_r += [each]


            # 3. sort second order relationships by value
            # --------------------------------------------
            sec_order_r = list(reversed(sorted(sec_order_r, key=lambda x: x[2])))

            # used for ending the loop
            # ------------------------
            change_flag += len(sec_order_r)
            
            ### debugging
            #print('first order')
            #print(first_order_r)
            
            #print('second order')
            #print(sec_order_r)

            # all good so far
            #################
            
            # 4. now to adjust 
            # ----------------
            for each_f in first_order_r:
                
                # init a first_order_flag
                # -----------------------
                first_order_flag = 1
                
                
                

                # making sure we pick the correct second order r
                # ----------------------------------------------
                for each_s in sec_order_r:
                    
                    # continue only if inside a VALID and not removed first order relationship
                    # using flag as cannot use break inside for and while
                    # ------------------------------------------------------------------------
                    if first_order_flag == 1:
                    
                        # 1. lists for adjustment
                        # init here
                        # ------------------------
                        new_first_order_to_add = []
                        first_order_to_remove = []
                        second_order_to_remove = []

                        ### debugging
                        #print('\n\n\n **********')
                        #print('first order relation effect before adjustment step: ')
                        #print(each_f)

                        #print('second order relation is:')
                        #print(each_s)



                        # check tree
                        # if the taker from first order relationship is the giver in second order relationship
                        # ------------------------------------------------------------------------------------
                        if each_f[1] == each_s[0]:

                            # receiver is taker
                            # based on value 
                            # can influence this edge - relationship
                            # --------------------------------------
                            if each_f[2] >= each_s[2]:

                                # 1. create new relationship
                                # this second order relation can be broken
                                # a new relationship is formed between given of first order & taker or sec order
                                # ------------------------------------------------------------------------------

                                # check if a relationship already exists and adjust that directly
                                # else add
                                # ---------------------------------------------------------------
                                rel_flag = 0
                                for each_r in r:
                                    if each_r[0] == each_f[0] and each_r[1] == each_s[1]:

                                        # relationship exists
                                        # -------------------
                                        rel_flag = 1

                                        # update value
                                        # ------------
                                        each_r[2] += each_s[2]

                                # do this only if relationship NOT present already
                                # ------------------------------------------------
                                if  rel_flag == 0:       
                                    new_first_order_to_add += [[each_f[0],each_s[1],each_s[2]]]

                                # remove second order from main anyways
                                # -------------------------------------
                                second_order_to_remove.append(each_s)


                                # 2. adjust existing first order relationship
                                # --------------------------------------------
                                if each_f[2] > each_s[2]:

                                    # adjusting existing first order relationship value directly
                                    # ----------------------------------------------------------
                                    each_f[2] = each_f[2] - each_s[2]

                                else:

                                    # looking like the values are equal meaning this relation can be removed
                                    # ----------------------------------------------------------------------
                                    first_order_to_remove.append(each_f)


                            else:

                                # check if a relationship already exists and adjust that directly
                                # else add
                                # ---------------------------------------------------------------
                                rel_flag = 0
                                for each_r in r:
                                    if each_r[0] == each_f[0] and each_r[1] == each_s[1]:

                                        # relationship exists
                                        # -------------------
                                        rel_flag = 1

                                        # update value
                                        # ------------
                                        each_r[2] += each_f[2]

                                # do this only if relationship NOT present already
                                # ------------------------------------------------
                                if  rel_flag == 0:       
                                    new_first_order_to_add  += [[each_f[0],each_s[1],each_f[2]]]


                                # this means that 1st order given giving less than what sec order taker owes
                                # here can remove first order relationship directly and adjust second order
                                # relationship accordingly
                                # --------------------------------------------------------------------------
                                first_order_to_remove.append(each_f)

                                # 1. adjust second order relationship
                                # -----------------------------------
                                each_s[2] = each_s[2] - each_f[2]


                            ### debugging
                            #print('new_first_order_to_add: ')
                            #print(new_first_order_to_add)

                            #print('first_order_to_remove: ')
                            #print(first_order_to_remove)

                            #print('second_order_to_remove: ')
                            #print(second_order_to_remove)

                            #print('current r before removal:')
                            #print(r)

                            # make adjustmensts to r here after ops so that
                            # it has effect immediately
                            ##
                            # now to perform updation operations on r
                            # just addition and removals
                            # ---------------------------------------
                            r += new_first_order_to_add

                            # remove second order relationships if any
                            # ---------------------------------------
                            for each in second_order_to_remove:
                                r.pop(r.index(each))

                            # remove first order relationships if any
                            # IF SO THEN BREAK SECOND ORDER LOOP
                            # ---------------------------------------
                            #for each in first_order_to_remove:
                            if len(first_order_to_remove) > 0:
                                
                                try:
                                    
                                    # having this is try just to manage circular reference
                                    # one off case like :
                                    # p1-5-p2, p2-5-p3, p3-5-p1 
                                    ######################################################
                                    
                                    # remove the first order relationship
                                    # which means cannot iter through remaning
                                    # second order relations
                                    # so break for and move to next first order relationship
                                    # -------------------------------------------------------
                                    r.pop(r.index(first_order_to_remove[0]))
                                    ### debugging
                                    #print('current r AFTER removal:')
                                    #print(r)
                                    first_order_flag = 0
                                
                                except:
                                    
                                    pass
                                


            
            # DONE WITH PER PERSON OPS
            ##########################
            ##########################
            ##########################
        
        
        
        # 2. outside while sanity priniting
        # ---------------------------------
        #print('at step ' + str(steps) + '..', end = '\r')
        
        # 2.
        # OUTSIDE PER PERSON FOR LOOP
        # inside while
        # check and break while
        # ----------------------------
        if change_flag == 0:
            
            # means iterating of all done and there are no second order relationships AT ALL
            # ------------------------------------------------------------------------------
            #print('done.')
            break
    
    
    # final ops
    # ---------
    net_post_dict = {}
    total_cash_flow_post = 0

    for each in r:

        # 1. add incoming
        # ----------------
        try:
            net_post_dict[each[1]] += each[2]
        except:
            net_post_dict[each[1]] = each[2]

        # 2. removing from payer
        # ----------------------
        try:
            net_post_dict[each[0]] -= each[2]
        except:
            net_post_dict[each[0]] = -1 * each[2]
        
        # 3. total value
        # --------------
        total_cash_flow_post += each[2]
    
    
    # print ops
    # ---------
    print('Number of times through all persons: ' + str(steps) + '\n')
    
    print('Before optimization stats - ')
    print('----------------------------- ')
    print('Total value flow before optimization: ' + str(total_cash_flow_orig))
    print('Net values before optimization - ')
    s_keys = list(sorted(net_dict.keys()))
    for each in s_keys:
        print(each + ': ' + str(net_dict[each]))
    
    
    print('\n******\n')
    print('After optimization stats - ')
    print('----------------------------- ')
    print('Total value flow after optimization: ' + str(total_cash_flow_post))
    print('Net values after optimization - ')
    for each in s_keys:
        print(each + ': ' + str(net_post_dict[each]))
    
    print('\n******\n')
    print('Value flow decorated: ')
    print('--------------------- ')
    for i in range(len(r)):
        print('step ' + str(i+1) + ': ' + str(r[i][0]) + ' gives ' + str(r[i][2]) + ' to ' + str(r[i][1]))
    
    
    
    # 3. outside WHILE loop
    # FINAL RETURN -- NOTING FOR NOW
    # -------------------------------
    #return r
     


# ### 2. Given a graph with N vertices and M edges along with the vertex pairs (vi, vj) and wi denoting the edges ((1,2), (1,3), (2,3) etc) and their weights, find the shortest path from the first vertex to all other vertices.


# function that is self explanatory with inline comments
# ------------------------------------------------------

def shortest_path(n, origin_node):
    
    '''
    
    1. simple weighted shortest path algorithm
    2. ENSURE origin_node IS CONNECTED TO ALL OTHER NODES ELSE THIS WILL THROW AN ERROR
    
    
    '''

    # 0. inits
    # --------
    shortest = {}
    parents = {}
    counter = 0

    # 0.1 need for while condition
    # ----------------------------
    all_nodes = []
    for each in n:
        if each[0] != origin_node:
            all_nodes.append(each[0])
        if each[1] != origin_node:
            all_nodes.append(each[1])
    all_nodes = sorted(list(set(all_nodes)))    


    # 1. init parent with origin node
    # -------------------------------
    for each in n:
        if each[0] == origin_node:
            parents[each[1]] = each[2]

    # 2. sorting by smaller value first
    # ---------------------------------
    parents = dict(sorted(parents.items(), key=lambda item: item[1]))

    # 2. while loop
    # loop until all nodes are in shortest
    # ------------------------------------
    while sorted(list(shortest.keys())) != sorted(all_nodes):

        # basically repeat this until all keys in shortest
        # ------------------------------------------------
        counter += 1

        # debugging
        ##
        #print('at counter: ' + str(counter))
        #print('shortest: ')
        #print(shortest)
        #print('parents: ')
        #print(parents)

        # 1. move first value - smallest node to shortest
        # -----------------------------------------------
        f_node = list(parents.keys())[0]
        f_value = parents[f_node]

        # 2. move f_node to shortest if it NOT present
        # ideally shouldnt be here
        # --------------------------------------------
        if f_node not in list(shortest.keys()):
            shortest[f_node] = f_value

        # 3. find immediate children of f_node and add to parents with UPDATED values
        # and ensure to keep the smaller one on parents
        # ----------------------------------------------------------------------------
        f_node_children = {}
        for each in n:
            if each[0] == f_node:

                # add child of relationship with 
                # 1. added value
                # 2. incase child node already there in parent - add if the added value LESS than already
                # existing value
                # AND ADD ONLY IF child_node NOT IN SHORTEST PATH
                # ----------------------------------------------------------------------------------------
                child_node = each[1]
                child_value = f_value + each[2]

                # adding by conditions stated above
                # ---------------------------------
                if child_node in list(parents.keys()):

                    # child node in parent already
                    # update the parent value ONLY if current value smaller
                    # -----------------------------------------------------
                    if parents[child_node] > child_value:
                        parents[child_node] = child_value

                else:

                    # child node not in parents
                    # add by conditions
                    # -------------------------
                    if child_node not in list(shortest.keys()):
                        parents[child_node] = child_value

        # 4. an update to parents
        # -----------------------
        del parents[f_node]
        parents = dict(sorted(parents.items(), key=lambda item: item[1]))
    
    
    
    # 3. final return
    # ---------------
    print(shortest)
    #return shortest


# ### helper funtions

# In[4]:


# simple wrapper
# --------------
def main_wrapper(task, input_in, origin_in):
    
    # simple if else
    # --------------
    if task == 'flow':
        cash_flow(input_in)
    else:
        shortest_path(input_in, origin_in)
        


# In[5]:


# simple assert
# -------------
def assert_args(args_in):

    # 0. make sure these are inline
    # -----------------------------
    assert args_in.task == 'flow' or args_in.task == 'path', 'Task Input Error: Task has to be flow or path'
    list_input = list(eval(str(args_in.input)))

    # 1. all good - next steps
    # ------------------------
    main_wrapper(args_in.task,list_input,args_in.origin_node)


# In[6]:


# end of functions
# parser ops
# ----------------

# 1. init parser
# ---------------
parser = argparse.ArgumentParser()

# 2. add argumenst to parser
# --------------------------
parser.add_argument('-t', '--task', help="Enter task to peform - 'flow': Minimize value flow between people, 'path': Shortest path from origin")
parser.add_argument('-i', '--input', help="Enter input for task. Both task follow the same format - relation is a list of lists in format [giver/from_node, taker_to_node, value/edge_weight] [...] [...] - example '['a','b',4], ['a','j',1]' -- PLEASE PASS THIS AS A STRING.")
parser.add_argument('-o', '--origin_node', help="Enter origin node for the shortest path task. Make sure origin is connected to all other nodes.")
# more args

# 3. parse arguments
# The arguments are parsed with parse_args(). 
# The parsed arguments are present as object attributes. 
# In our case, there will be args.task,  args.file_url & args.input_mode attribute.
# ---------------------------------------------------------------------------------
args = parser.parse_args()

# 4. CALLING MAIN FUNCTION
###########################
assert_args(args)
