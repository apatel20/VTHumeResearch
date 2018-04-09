import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

number_train, number_test = X_train.shape[0], X_test.shape[0]

number_classes = np.unique(y_train).shape[0]#getting number of unique classes

train_distribution, test_distribution = np.zeros(number_classes), np.zeros(number_classes)
for class in range(nunber_classes)
	train_distribution[class] = np.sum(y_train == class) / number_train
	test_distribution[class] = np.sum(y_test == class) / number_test

#setting the plot features and finally graphing the data set
fig, ax = plt.subplots
column_width = 0.5
bar_train = ax.bar(np.arange(number_classes), train_distribution, width=column_width, color='r')
bar_test = ax.bar(np.arange(number_classes)+column_width, test_distribution, width=column_width, color='b')
ax.set_ylabel('Percentage of total data set')
ax.set_xlabel('CLASS #')
ax.set_title('Classes distribution in traffic sign dataset')
ax.set_xticks(np.arange(0, number_classes, 5)+col_width)
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, number_classes, 5)])
ax.legend((bar_train[0], bar_test[0]), ('training dataset', 'testing dataset'))
plt.show()