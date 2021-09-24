import numpy as np
import random

def huffman_tree(nodes):
    '''
    nodes : node={weight:12,name:"我"}
    '''
    if len(nodes) > 1:
        min_idx = np.argmin([n["weight"] for n in nodes])
        second_min_idx = np.argmin([n["weight"] if i!=min_idx else 9999999999 for i,n in enumerate(nodes)])
        new_nodes = [n for i,n in enumerate(nodes) if i not in [min_idx,second_min_idx]]
        node_i,node_j = nodes[min_idx],nodes[second_min_idx]
        node = {"weight":node_i["weight"]+node_j["weight"],"name":node_i["name"]+"+"+node_j["name"],"left_subnode":node_j,"right_subnode":node_i}
        return huffman_tree(new_nodes+[node])
    return nodes

def print_tree(tree,t):
    if tree:
        print("\t"*t,tree["name"]+":"+str(tree["weight"]))
        if "left_subnode" in tree.keys():
            print_tree(tree["left_subnode"],t+1)
        if "right_subnode" in tree.keys():
            print_tree(tree["right_subnode"],t+1)

if __name__ == "__main__":
    nodes = [{"weight":10,"name":"我"},{"weight":8,"name":"你"},{"weight":3,"name":"爱"},{"weight":4,"name":"中国"},{"weight":20,"name":"在"},{"weight":12,"name":"哪里"}]
    tree = huffman_tree(nodes)[0]
    print_tree(tree,1)
