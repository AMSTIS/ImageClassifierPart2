import argparse
import torch

from collections import OrderedDict
from torchvision import datasets
from torchvision import transforms
from torchvision import models


def get_transformed_validation_data_from_dir(directory):
    """
    Performs training transformations on a dataset
    """
    composed_transformations = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return datasets.ImageFolder(directory, transform=composed_transformations)


def get_transformed_training_data_from_dir(directory):
    """
    Performs test/validation transformations on a dataset
    """
    composed_transformations = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return datasets.ImageFolder(directory, transform=composed_transformations)


def get_data_loader(transformed_data, is_training_data=True):
    """
    Creates and returns a data loader from transformed_data
    """
    return torch.utils.data.DataLoader(transformed_data, batch_size=50, shuffle=True) if is_training_data else torch.utils.data.DataLoader(transformed_data, batch_size=50)


def get_pre_trained_model(architecture="vgg16"):
    """
    Returns the pre trained pre_trainined_model.
    """
    pre_trainined_model = models.vgg16(pretrained=True)
    pre_trainined_model.name = architecture

    for model_parameter in pre_trainined_model.parameters():
        model_parameter.requires_grad = False
    return pre_trainined_model


def create_classifier(model, dnn_units):
    """
    Creates a classifier with the corect number of input layers
    """
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(input_features, dnn_units, bias=True)),
                                                  ('relu1', torch.nn.ReLU()),
                                                  ('dropout1', torch.nn.Dropout(p=0.5)),
                                                  ('fc2', torch.nn.Linear(dnn_units, 102, bias=True)),
                                                  ('output', torch.nn.LogSoftmax(dim=1))]))
    return classifier


def perform_validation(pre_trained_model, loader_for_test_data, criterion, torch_device):
    """
    Validates training against testloader to return loss and accuracy
    """
    loss_incurred_during_test = 0
    accuracy = 0
    
    for index, (inputs, labels) in enumerate(loader_for_test_data):
        inputs = inputs.to(torch_device)
        labels = labels.to(torch_device)
        
        pre_trained_model_output = pre_trained_model.forward(inputs)
        loss_incurred_during_test += criterion(pre_trained_model_output, labels).item()
        
        pre_trained_model_output_exp = torch.exp(pre_trained_model_output)
        equality = (labels.data == pre_trained_model_output_exp.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return loss_incurred_during_test, accuracy


def re_train_model(pre_trained_model, loader_for_training_data, loader_for_test_data, torch_device, criterion, optimizer, epoch_value, output_interval, steps):
    """
    Performs training of the given model.
    """
    print("[ Training Started ]")
    print("")

    epoch_value = epoch_value if epoch_value else 5
    for current_epoch in range(epoch_value):
        current_loss = 0

        for index, (inputs, labels) in enumerate(loader_for_training_data):
            steps += 1
            
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = pre_trained_model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            current_loss += loss.item()
        
            if steps % output_interval == 0:
                pre_trained_model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = perform_validation(pre_trained_model, loader_for_test_data, criterion, torch_device)

                print("Current step/epoch value: " + str(current_epoch+1) + "/" + str(epoch_value))
                print("Incurred loss during training: " + str(current_loss/output_interval))
                print("Current step/epoch value: " + str(validation_loss/len(loader_for_test_data)))
                print("Current step/epoch value: " + str(accuracy/len(loader_for_test_data)))

                current_loss = 0
                pre_trained_model.train()

    print("Training process End .....\n")
    return pre_trained_model


def validate_model(pre_trained_model, loader_for_test_data, torch_device):
    """
    Validates the model on test data images.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        pre_trained_model.eval()
        for data in loader_for_test_data:
            images, labels = data
            images, labels = images.to(torch_device), labels.to(torch_device)
            outputs = pre_trained_model(images)
            temp_val, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Achieved accuracy by the model on test images is: ' + str(float(100 * correct / total)))


def save_state_of_model(pre_trained_model, transformed_data_for_training):
    """
    Saves the model at a defined checkpoint
    """
    pre_trained_model.class_to_idx = transformed_data_for_training.class_to_idx

    checkpoint = {'architecture': pre_trained_model.name,
                  'classifier': pre_trained_model.classifier,
                  'class_to_idx': pre_trained_model.class_to_idx,
                  'state_dict': pre_trained_model.state_dict()}

    torch.save(checkpoint, 'my_checkpoint.pth')


def start_main():
    argument_parser_instance = argparse.ArgumentParser()
    argument_parser_instance.add_argument('--rate_of_learning', type=float, help='Rate of learning for the pre_trained_model')
    argument_parser_instance.add_argument('--dnn_units', type=int, help='Value of units for classifier')
    argument_parser_instance.add_argument('--epoch_value', type=int, help='Value of epoch for training')
    parsed_arguments = argument_parser_instance.parse_args()

    directory_name_of_stored_data = 'flowers'
    directory_for_training_data = directory_name_of_stored_data + '/train'
    directory_for_validation_data = directory_name_of_stored_data + '/valid'
    directory_for_test_data = directory_name_of_stored_data + '/test'

    transformed_data_for_training = get_transformed_training_data_from_dir(directory_for_training_data)
    transformed_data_for_validation = get_transformed_validation_data_from_dir(directory_for_validation_data)
    transformed_data_for_testing = get_transformed_validation_data_from_dir(directory_for_test_data)
    
    loader_for_training_data = get_data_loader(transformed_data_for_training)
    loader_for_validation_data = get_data_loader(transformed_data_for_validation, is_training_data=False)
    loader_for_test_data = get_data_loader(transformed_data_for_testing, is_training_data=False)

    pre_trained_model = get_pre_trained_model("vgg16")

    pre_trained_model.classifier = create_classifier(pre_trained_model, dnn_units=parsed_arguments.dnn_units if parsed_arguments.dnn_units else 4096)
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pre_trained_model.to(torch_device)

    rate_of_learning = parsed_arguments.rate_of_learning if parsed_arguments.rate_of_learning else 0.025

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(pre_trained_model.classifier.parameters(), lr=rate_of_learning)

    output_interval = 30
    steps = 0
    re_trained_model = re_train_model(pre_trained_model, loader_for_training_data, loader_for_validation_data,
                                   torch_device, criterion, optimizer, parsed_arguments.epoch_value,
                                   output_interval, steps)
    
    print("")
    print("[ Training Complete ]")

    validate_model(re_trained_model, loader_for_test_data, torch_device)
    save_state_of_model(re_trained_model, transformed_data_for_training)


"""
This is done so the main() method is not executed if this script is imported from another module.
"""
if __name__ == '__main__':
    start_main()
