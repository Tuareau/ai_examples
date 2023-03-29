import tuai
import lrmodel

def lrmodel_example(center0, center1):

    data_class0 = tuai.generate_set_of_data(100, 0, center0)
    data_class1 = tuai.generate_set_of_data(100, 1, center1)
    train_data_class0, test_data_class0 = tuai.split_train_test(data_class0, 0.2)
    train_data_class1, test_data_class1 = tuai.split_train_test(data_class1, 0.2)
    train_data = train_data_class0 + train_data_class1
    test_data = test_data_class0 + test_data_class1

    tuai.show_data(train_data, test_data)

    lrmodel.visualize_logistic_regression(train_data, test_data)

#lrmodel_example([2,2], [1,1])

#lrmodel_example([2,2], [0,0])

lrmodel_example([2,2], [2,2])