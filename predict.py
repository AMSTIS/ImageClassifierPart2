import argparse
import json
import math

import numpy
import torch
import PIL
from torchvision import models


def load_pre_trained_model_state(saved_state_file_name):
    """
    Loads our saved deep learning pre_trained_model from loaded_state.
    """
    saved_state_file_name = saved_state_file_name if saved_state_file_name else "my_checkpoint.pth"
    loaded_state = torch.load(saved_state_file_name)

    pre_trained_model = models.vgg16(pretrained=True)
    pre_trained_model.name = "vgg16"

    for param in pre_trained_model.parameters():
        param.requires_grad = False

    pre_trained_model.class_to_idx = loaded_state['class_to_idx']
    pre_trained_model.classifier = loaded_state['classifier']
    pre_trained_model.load_state_dict(loaded_state['state_dict'])
    
    return pre_trained_model


def process_selected_image_file(image_path):
    """
    Performs cropping, scaling of image for our model
    """
    selected_image = PIL.Image.open(image_path)
    image_width, image_height = selected_image.size

    if image_width < image_height:
        modeified_size = [256, 256**600]
    else:
        modeified_size = [256**600, 256]
        
    selected_image.thumbnail(size=modeified_size)

    # Find pixels to crop on to create 224x224 image
    image_center_coordinates = image_width/4, image_height/4
    image_left_coordinates = image_center_coordinates[0]-(244/2)
    image_top_coordinates = image_center_coordinates[1]-(244/2)
    image_right_coordinates = image_center_coordinates[0]+(244/2)
    image_bottom_coordinates = image_center_coordinates[1]+(244/2)
    selected_image = selected_image.crop((image_left_coordinates, image_top_coordinates,
                                          image_right_coordinates, image_bottom_coordinates))

    numpy_processed_image = numpy.array(selected_image)/255
    normalize_means = [0.485, 0.456, 0.406]
    normalize_standard_deviation = [0.229, 0.224, 0.225]
    numpy_processed_image = (numpy_processed_image-normalize_means)/normalize_standard_deviation
    numpy_processed_image = numpy_processed_image.transpose(2, 0, 1)

    return numpy_processed_image


def try_predicting(image_tensor, pre_trained_model, torch_device, category_to_name_dict, top_k_value=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    pre_trained_model.eval()

    torch_image = torch.from_numpy(numpy.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)
    pre_trained_model = pre_trained_model.cpu()

    probability = pre_trained_model.forward(torch_image)
    linear_probability = torch.exp(probability)

    probability_top_value, probability_top_labels = linear_probability.topk(top_k_value)
    probability_top_value = numpy.array(probability_top_value.detach())[0]
    probability_top_labels = numpy.array(probability_top_labels.detach())[0]

    idx_to_class = {val: key for key, val in pre_trained_model.class_to_idx.items()}
    probability_top_labels = [idx_to_class[lab] for lab in probability_top_labels]
    identified_top_flowers = [category_to_name_dict[lab] for lab in probability_top_labels]
    
    return probability_top_value, probability_top_labels, identified_top_flowers


def output_result(probs, flowers):
    for row, col in enumerate(zip(flowers, probs)):
        print("Row :" + str(row+1))
        print("Flower :" + str(col[1]))
        print("Match :" + str(math.ceil(col[0]*100)))


def start_main():
    argument_parser_instance = argparse.ArgumentParser()
    argument_parser_instance.add_argument('--selected_image', type=str, help='Selected image for prediction.', required=True)
    argument_parser_instance.add_argument('--existing_saved_state', type=str, help='File to load pre_trained_model\'s state.', required=True)
    argument_parser_instance.add_argument('--categories_mapping', type=str, help='File for categories mapping.')
    parsed_arguments = argument_parser_instance.parse_args()    

    with open(parsed_arguments.categories_mapping, 'r') as f:
        category_to_name_dict = json.load(f)

    pre_trained_model = load_pre_trained_model_state(parsed_arguments.existing_saved_state)
    image_tensor = process_selected_image_file(parsed_arguments.selected_image)
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probability_top_value, top_labels, identified_top_flowers = try_predicting(image_tensor, pre_trained_model, torch_device, category_to_name_dict)

    output_result(identified_top_flowers, probability_top_value)


"""
This is done so the main() method is not executed if this script is imported from another module.
"""
if __name__ == '__main__':
    start_main()
