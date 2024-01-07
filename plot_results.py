from matplotlib import pyplot as plt
import json

def plot_highest(filename, plot_target):
    with open(f'{filename}.json', 'r') as f:
        data = json.load(f)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    flags = [False, False, False]
    match plot_target:
        case 'lr':
            highest_training_2 = None
            highest_testing_2 = None
            highest_training_3 = None
            highest_testing_3 = None
            highest_training_4 = None
            highest_testing_4 = None
            for line in data:
                match line['learning_rate']:
                    case 0.01:
                        if highest_training_2 == None:
                            highest_training_2 = line['training_accuracies']
                            highest_testing_2 = line['testing_averages']
                        else:
                            #if highest_training_2[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_2[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_2 = line['testing_averages']
                                highest_training_2 = line['training_accuracies']
                    case 0.001:
                        if highest_training_3 == None:
                            highest_training_3 = line['training_accuracies']
                            highest_testing_3 = line['testing_averages']
                        else:
                            #if highest_training_3[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_3[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_3 = line['testing_averages']
                                highest_training_3 = line['training_accuracies']
                    case 0.0001:
                        if highest_training_4 == None:
                            highest_training_4 = line['training_accuracies']
                            highest_testing_4 = line['testing_averages']
                        else:
                            #if highest_training_4[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_4[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_4 = line['testing_averages']
                                highest_training_4 = line['training_accuracies']

            plt.plot(highest_training_2, label=f'lr0.01', color='blue')
            for test in highest_testing_2:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='blue')
            plt.plot(highest_training_3, label=f'lr0.001', color='red')
            for test in highest_testing_3:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='red')
            plt.plot(highest_training_4, label=f'lr0.0001', color='green')
            for test in highest_testing_4:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='green')

        case 'neurons':
            highest_training_64 = None
            highest_testing_64 = None
            highest_training_128 = None
            highest_testing_128 = None
            highest_training_256 = None
            highest_testing_256 = None
            for line in data:
                match line['neuron_value']:
                    case 64:
                        if highest_training_64 == None:
                            highest_training_64 = line['training_accuracies']
                            highest_testing_64 = line['testing_averages']
                        else:
                            #if highest_training_2[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_64[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_64 = line['testing_averages']
                                highest_training_64 = line['training_accuracies']
                    case 128:
                        if highest_training_128 == None:
                            highest_training_128 = line['training_accuracies']
                            highest_testing_128 = line['testing_averages']
                        else:
                            #if highest_training_3[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_128[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_128 = line['testing_averages']
                                highest_training_128 = line['training_accuracies']
                    case 256:
                        if highest_training_256 == None:
                            highest_training_256 = line['training_accuracies']
                            highest_testing_256 = line['testing_averages']
                        else:
                            #if highest_training_4[-1] < line['training_accuracies'][-1]:
                                
                            if highest_testing_256[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_256 = line['testing_averages']
                                highest_training_256 = line['training_accuracies']

            plt.plot(highest_training_64, label=f'n64', color='blue')
            for test in highest_testing_64:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='blue')
            plt.plot(highest_training_128, label=f'n128', color='red')
            for test in highest_testing_128:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='red')
            plt.plot(highest_training_256, label=f'n256', color='green')
            for test in highest_testing_256:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='green')


        case 'dropout':
            highest_training_08 = None
            highest_testing_08 = None
            highest_training_1 = None
            highest_testing_1 = None
            for line in data:
                match line['dropout_value']:
                    case 0.8:
                        if highest_training_08 == None:
                            highest_training_08 = line['training_accuracies']
                            highest_testing_08 = line['testing_averages']
                        else:
                            if highest_testing_08[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_08 = line['testing_averages']
                                highest_training_08 = line['training_accuracies']
                    case 1:
                        if highest_training_1 == None:
                            highest_training_1 = line['training_accuracies']
                            highest_testing_1 = line['testing_averages']
                        else:     
                            if highest_testing_1[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_1 = line['testing_averages']
                                highest_training_1 = line['training_accuracies']

            plt.plot(highest_training_08, label=f'd0.8', color='blue')
            for test in highest_testing_08:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='blue')
            plt.plot(highest_training_1, label=f'd1', color='red')
            for test in highest_testing_1:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='red')

        case 'reg':
            highest_training_l1 = None
            highest_testing_l1 = None
            highest_training_l2 = None
            highest_testing_l2 = None
            for line in data:
                match line['regularizer']:
                    case 'l1':
                        if highest_training_l1 == None:
                            highest_training_l1 = line['training_accuracies']
                            highest_testing_l1 = line['testing_averages']
                        else:
                            if highest_testing_l1[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_l1 = line['testing_averages']
                                highest_training_l1 = line['training_accuracies']
                    case 'l2':
                        if highest_training_l2 == None:
                            highest_training_l2 = line['training_accuracies']
                            highest_testing_l2 = line['testing_averages']
                        else:     
                            if highest_testing_l2[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_l2 = line['testing_averages']
                                highest_training_l2 = line['training_accuracies']

            plt.plot(highest_training_l1, label=f'l1', color='blue')
            for test in highest_testing_l1:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='blue')
            plt.plot(highest_training_l2, label=f'l2', color='red')
            for test in highest_testing_l2:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='red')

        case 'activation':
            highest_training_relu = None
            highest_testing_relu = None
            highest_training_sig = None
            highest_testing_sig = None
            for line in data:
                match line['activation_function']:
                    case 'relu':
                        if highest_training_relu == None:
                            highest_training_relu = line['training_accuracies']
                            highest_testing_relu = line['testing_averages']
                        else:
                            if highest_testing_relu[-1]['test_average'] < line['testing_averages'][-1]['test_average']:
                                highest_testing_relu = line['testing_averages']
                                highest_training_relu = line['training_accuracies']
                    case 'sigmoid':
                        if highest_training_sig == None:
                            highest_training_sig = line['training_accuracies']
                            highest_testing_sig = line['testing_averages']
                        else:     
                            if highest_training_sig[-1] < line['training_accuracies'][-1]:
                                highest_testing_sig = line['testing_averages']
                                highest_training_sig = line['training_accuracies']

            plt.plot(highest_training_relu, label=f'relu', color='blue')
            for test in highest_testing_relu:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='blue')
            plt.plot(highest_training_sig, label=f'sigmoid', color='red')
            for test in highest_testing_sig:
                plt.plot(test['epoch'], test['test_average'], marker='x', color='red')

    plt.legend()
    plt.show()

plot_highest('task1-all-results', 'lr')
plot_highest('task1-all-results', 'neurons')
plot_highest('task1-all-results', 'dropout')
plot_highest('task1-all-results', 'reg')
plot_highest('task1-all-results', 'activation')