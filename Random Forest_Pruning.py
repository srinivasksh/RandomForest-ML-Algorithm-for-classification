import csv
import random
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# Implement your decision tree below
class DecisionTree():
	
	def learn(self, training_set,features):
        # implement this function
		
		def split_tree(attr,val,ipdata):
			tree_left = []
			tree_right = []
			for ix in ipdata:
				if float(ix[attr]) < float(val):
					tree_left.append(ix)
				else:
					tree_right.append(ix)
			return tree_left,tree_right
		
		def calc_entropy(ip_vec):
			ix0_count = 0
			ix1_count = 0
			for ix in ip_vec:
				if int(ix) == 0:
					ix0_count += 1
				else:
					ix1_count += 1
			if ix0_count == 0 or ix1_count == 0:
				return 0
			else:
				return (-1*(((ix0_count/(ix0_count+ix1_count))*(np.log2((ix0_count/(ix0_count+ix1_count)))))+((ix1_count/(ix0_count+ix1_count))*(np.log2(ix1_count/(ix0_count+ix1_count))))))		
	
		def calc_info_gain(parent_ent,i,tree_lt,tree_rt):
			lt_entropy = calc_entropy([ix[-1] for ix in tree_lt])
			rt_entropy = calc_entropy([ix[-1] for ix in tree_rt])
			tmp_gain = parent_ent - ((len(tree_lt)/(len(tree_lt)+len(tree_rt)))*lt_entropy) - ((len(tree_rt)/(len(tree_lt)+len(tree_rt)))*rt_entropy)
			return tmp_gain
		
		def chk_last_node(ipSet):
			class_val = [elem[-1] for elem in ipSet]
			response = max(set(class_val),key=class_val.count)
			return(response)
			
		
		def get_child(ipdata,features):
		
			def sortByIndex(item):
				return item[2]
				
			decision_cols = random.sample(list(range(len(ipdata[0])-1)),features)
			parent_entropy = calc_entropy([ip[-1] for ip in ipdata])
			info_gain = []
			
			for ix in decision_cols:
				ix_values = set(ix1[ix] for  ix1 in ipdata)
				for ix_val in ix_values:
					result_left, result_right = split_tree(ix,ix_val,ipdata)
					info_gain.append((ix,ix_val,calc_info_gain(parent_entropy,ix,result_left,result_right),result_left,result_right))
			
			info_gain=sorted(info_gain,key=sortByIndex,reverse=True)
			return {'attribute':info_gain[0][0],'chk_value':info_gain[0][1],'left_branch':info_gain[0][3],'right_branch':info_gain[0][4]}
		
		def splitTree(main_tree,features):
			left_child = main_tree['left_branch']
			right_child = main_tree['right_branch']

			del(main_tree['left_branch'])
			del(main_tree['right_branch'])
			
			if not left_child and right_child:
				main_tree['left_child'] = chk_last_node(right_child)
				main_tree['right_child'] = chk_last_node(right_child)
				return
			
			if not right_child and left_child:
				main_tree['left_child'] = chk_last_node(left_child)
				main_tree['right_child'] = chk_last_node(left_child)
				return

			main_tree['left_child'] = get_child(left_child,features)
			splitTree(main_tree['left_child'],features)
				
			main_tree['right_child'] = get_child(right_child,features)
			splitTree(main_tree['right_child'],features)
				
			return main_tree
		
		## Start of the functin
		self.tree = {}
		self.tree = get_child(training_set,features)
		splitTree(self.tree,features)
	

    # implement this function
	def classify(self, test_instance):
		result = 0 # baseline: always classifies as 0
		
		def get_class(tree_input,test_data):
			if float(test_data[tree_input['attribute']]) < float(tree_input['chk_value']):
				if isinstance(tree_input['left_child'], dict):
					return get_class(tree_input['left_child'],test_data)
				else:
					return tree_input['left_child']
			else:
				if isinstance(tree_input['right_child'], dict):
					return get_class(tree_input['right_child'],test_data)
				else:
					return tree_input['right_child']
		
		result = get_class(self.tree,test_instance)
		return result

# Load data set
with open("spam.data") as f:
    data = [tuple(line) for line in csv.reader(f, delimiter=" ")]
ip_size = len(data)
print("Number of records in input file: %d" % ip_size)

## Split data into train and test
random.shuffle(data)
X_train = data[:round(len(data)*0.7)]
X_test = data[round(len(data)*0.7):]

def run_random_forest(num_of_trees,num_of_features):
	tree = [None]*num_of_trees
	train_keys = [None]*num_of_trees
	oob_keys = [None]*num_of_trees

	for ix in list(range(num_of_trees)):
		sample_ix = np.random.choice(len(X_train)-1,len(X_train),replace=True)
		train_keys[ix] = list(set(sample_ix))
		oob_keys[ix] = list(set(list(range(len(X_train)))) - set(sample_ix))
		training_set = []
		for i in sample_ix:
			training_set.append(X_train[i])
		tree[ix] = DecisionTree()
		
		# Construct a tree using training set, set maximum depth to prevent looping
		output = tree[ix].learn(training_set,num_of_features)

	# Classify the test set using the tree we just constructed
	results = []
	for instance in X_test:
		tree_output = []
		for ix in list(range(num_of_trees)):
			result = tree[ix].classify( instance[:-1] )
			tree_output.append(result)
		spam_result = max(set(tree_output),key=tree_output.count)
		results.append( spam_result == instance[-1])

	# Accuracy
	test_error = 1-(float(results.count(True))/float(len(results)))
	
	## OOB Error
	oob_results = []
	oob_keys = list(set([item for sublist in oob_keys for item in sublist]))
	for ip_key in oob_keys:
		accuracy_oob = []
		for ix in list(range(num_of_trees)):
			if ip_key not in train_keys[ix]:
				result = tree[ix].classify( training_set[ip_key][:-1] )
				accuracy_oob.append(result)
		spam_result = max(set(accuracy_oob),key=accuracy_oob.count)
		oob_results.append( spam_result == training_set[ip_key][-1])
		
	## OOB Accuracy
	oob_error = 1-(float(oob_results.count(True))/float(len(oob_results)))
	return (test_error,oob_error)

if __name__ == "__main__":
	num_of_trees = 20
	n_features = [1,5,7,9,11,15]
	native_results = []
	sklearn_results = []
	test_error_list = []
	oob_error_list = []
	for num_of_features in n_features:
		print(" ")
		print("Building Random Forest with " + str(num_of_features) + " features...")
		test_error,oob_error = run_random_forest(num_of_trees,num_of_features)
		test_error_list.append(test_error)
		oob_error_list.append(oob_error)
		print("Done!")
	
	plt.plot(n_features, test_error_list, 'x',color='green',label="Test Error")
	plt.plot(n_features, oob_error_list, 'o',color='red',label="OOB Error")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()