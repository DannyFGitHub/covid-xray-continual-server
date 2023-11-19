import queue
import datetime
import os
import json
import random
from typing import Sequence
import numpy as np
from PIL import Image
from torch.autograd import Variable
import threading
import pytz

import matplotlib.pyplot as plt
import seaborn as sn
import torch
from avalanche.benchmarks.generators import ni_benchmark, filelist_benchmark, tensors_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
    confusion_matrix_metrics, bwt_metrics, gpu_usage_metrics, forward_transfer_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.models import SimpleCNN
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
from avalanche.training.supervised import Naive, Replay, GDumb, Cumulative, LwF, AGEM, EWC, \
    SynapticIntelligence, PNNStrategy, GEM
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import time

# For downloading images
import requests
from io import BytesIO

from firebase import get_submissions_from_firestore, set_prediction_in_firestore, set_learnt_in_firestore, set_prediction_in_test_batch_firestore, get_submissions_from_firestore_with_date

from dotenv import dotenv_values
dotenv_config = dotenv_values(".env")
# Queue is thread safe

# Data Source Class for loading the directory information


class DataSource:

    # Prefix the model name with a random session id for easier grouping whenever model name is used.
    def set_random_session_id(self):
        random_session_id = random.randint(1111, 9999)
        self.candidate_model_name = str(
            random_session_id) + "_" + self.candidate_model_name

    def __init__(self):
        self.train_data_dir = ""
        self.test_data_dir = ""
        self.candidate_model_name = ""

    def CLCOVIDXRAY_balanced_unsegmented_live_train(self):
        self.train_data_dir = "./Datasets/CLCOVIDXRAY-balanced-unsegmented_live_train/train"
        self.test_data_dir = "./Datasets/CLCOVIDXRAY-balanced-unsegmented_live_train/test"
        self.candidate_model_name = "CLCOVIDXRAY_balanced_unsegmented_live_train"
        self.set_random_session_id()

    # def CLCOVIDXRAY_balanced_unsegmented_500(self):
    #     self.train_data_dir = "./Datasets/CLCOVIDXRAY-balanced-unsegmented_500/train"
    #     self.test_data_dir = "./Datasets/CLCOVIDXRAY-balanced-unsegmented_500/test"
    #     self.candidate_model_name = "CLCOVIDXRAY_balanced_unsegmented_500"
    #     self.set_random_session_id()


class Confusion_Matrix_Maker:

    def __init__(self, initial_class_names):
        self.initial_class_names = initial_class_names
        self.model_name = COVID_XRAY_CL.CURRENT_GLOBAL_MODEL_NAME

    def custom_cm_image_creator(self,
                                confusion_matrix_tensor: Tensor,
                                display_labels: Sequence = None,
                                include_values=False,
                                xticks_rotation=0,
                                yticks_rotation=0,
                                values_format=None,
                                cmap='viridis',
                                image_title=''):
        # Custom Confusion Matrix
        sn.set(color_codes=True)
        fig = plt.figure(1, figsize=(9, 6))
        plt.title("Confusion Matrix")
        cm = confusion_matrix_tensor.numpy()
        sn.set(font_scale=1.4)
        confusion_matrix_seaborn = sn.heatmap(cm, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'},
                                              fmt='g')
        n_classes = cm.shape[0]
        display_labels = self.initial_class_names
        confusion_matrix_seaborn.set_xticklabels(display_labels)
        confusion_matrix_seaborn.set_yticklabels(display_labels)
        confusion_matrix_seaborn.set(
            ylabel="True Label", xlabel="Predicted Label")

        # Save figure
        plt.savefig(
            COVID_XRAY_CL.DESTINATION_FOLDER + self.model_name + "-" + datetime.datetime.now().isoformat() +
            "-confusion-matrix.pdf",
            format="pdf",
            bbox_inches="tight")

        return fig


class COVID_XRAY_CL:

    TIME_TESTS_IMAGE_LIMIT = 200

    # CONSTANTS
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    IMAGE_SIZE = 224
    # Minimum 30 epochs
    EPOCHS = 20
    NUM_EXPERIENCES = 1
    # EPOCHS = 1
    # NUM_EXPERIENCES = 1

    TRAIN_MINIBATCH_SIZE = 1
    EVAL_MINIBATCH_SIZE = 1

    # Only eval at the end of the experience
    EVAL_EVERY = 0

    # Increased learning rate for the continual learning live experiment in order to see bigger impact on training changes for visualization only
    # INITIAL_LEARNING_RATE = 0.000001
    INITIAL_LEARNING_RATE = 0.00001

    # used for access when cannot be passed
    CURRENT_GLOBAL_MODEL_NAME: str = ""

    # DEVICE CUDA or CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", DEVICE)

    DESTINATION_FOLDER = "./results/"
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)

    TB_LOG_DIR = "./tb_data"
    if not os.path.exists(TB_LOG_DIR):
        os.makedirs(TB_LOG_DIR)

    # Initiate the queue
    INPUT_QUEUE = queue.Queue()

    # Array for storing the results
    INFERENCE_TEST_RESULTS = []
    TRAINING_TEST_RESULTS = []

    @staticmethod
    def notifier_send(message: str, print_message: bool = False):
        print(message)

    @staticmethod
    def prepare_optimizer(model):
        # optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
        optimizer = Adam(model.parameters(
        ), lr=COVID_XRAY_CL.INITIAL_LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
        # Adam Recom settings: https://arxiv.org/abs/1412.6980 learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
        return optimizer

    @staticmethod
    def prepare_criterion():
        criterion = CrossEntropyLoss()
        return criterion

    @staticmethod
    def prepare_evaluator(num_classes, name_of_run, class_names):
        # DEFINE THE EVALUATION PLUGIN and LOGGERS
        # The evaluation plugin manages the metrics computation.
        # It takes as argument a list of metrics, collectes their results and returns
        # them to the strategy it is attached to.

        # log to Tensorboard
        tb_logger = TensorboardLogger(tb_log_dir=os.path.join(
            COVID_XRAY_CL.TB_LOG_DIR, name_of_run))

        # log to text file
        # text_logger = TextLogger(open('log.txt', 'a'))

        # print to stdout
        interactive_logger = InteractiveLogger()

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=False,
                             experience=True, stream=False),
            loss_metrics(minibatch=False, epoch=False,
                         experience=True, stream=False),
            timing_metrics(experience=True, epoch=False),
            forgetting_metrics(experience=True, stream=False),
            bwt_metrics(experience=True, stream=False),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(image_creator=Confusion_Matrix_Maker(class_names).custom_cm_image_creator,
                                     class_names=class_names,
                                     num_classes=num_classes, save_image=True, stream=True),
            forward_transfer_metrics(experience=True, stream=False),
            collect_all=True,  # this is default value anyway
            loggers=[interactive_logger, tb_logger],  # text_logger],
        )

        return eval_plugin

    @staticmethod
    def single_image_loader(transformer, image_path):
        """load image, returns cuda tensor"""

        # If image path is url, download it
        if image_path.startswith('http'):
            image = Image.open(BytesIO(requests.get(image_path).content))
        else:
            image = Image.open(image_path)

        image = image.convert("RGB")
        image = transformer(image).float()
        image = Variable(image, requires_grad=True)
        # this is for VGG, may not be needed for ResNet
        image = image.unsqueeze(0)

        if COVID_XRAY_CL.DEVICE.type == 'cuda':
            return image.cuda()  # assumes that you're using GPU
        else:
            return image

    @staticmethod
    def single_image_loader_to_tensor(transformer, image_path):
        """load image, returns cuda tensor"""
        # If image path is url, download it
        if image_path.startswith('http'):
            image = Image.open(BytesIO(requests.get(image_path).content))
        else:
            image = Image.open(image_path)

        image = image.convert("RGB")
        image_tensor = transformer(image)
        return image_tensor

    @staticmethod
    def get_train_transform():
        return Compose([
            # rotating flippping, https://pytorch.org/vision/stable/transforms.html
            Resize((COVID_XRAY_CL.IMAGE_SIZE, COVID_XRAY_CL.IMAGE_SIZE)),
            ToTensor(),  # Turn PIL Image to torch.Tensor
            Normalize(COVID_XRAY_CL.MEAN, COVID_XRAY_CL.STD)
        ])

    @staticmethod
    def get_eval_transform():
        return Compose([
            Resize((COVID_XRAY_CL.IMAGE_SIZE, COVID_XRAY_CL.IMAGE_SIZE)),
            ToTensor(),
            Normalize(COVID_XRAY_CL.MEAN, COVID_XRAY_CL.STD)
        ])

    @staticmethod
    def start_cl_engine(train_dir: str, test_dir: str, model, model_name, strategy_lambda: callable, strategy_name):
        # Define transformations for training
        train_transform = COVID_XRAY_CL.get_train_transform()

        # Define transformations for evaluation
        test_transform = COVID_XRAY_CL.get_eval_transform()

        train_dataset = ImageFolder(train_dir, transform=train_transform)
        test_dataset = ImageFolder(test_dir, transform=test_transform)

        # num_of_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        # class_names = ['COVID-19', 'Normal', 'Pneumonia']

        # notifier_send("Class Names:" + str(class_names), print_message=True)

        # Requires a reference structure dataset to work with, for testing purposes only.
        main_scenario = ni_benchmark(
            train_dataset=train_dataset, test_dataset=test_dataset, n_experiences=COVID_XRAY_CL.NUM_EXPERIENCES, shuffle=False, seed=1234,
            balance_experiences=True,
        )
        num_classes = main_scenario.n_classes

        # Classes passed manually and must be consistent with the way the foundation model was trained.

        # Prepare Optimiser
        optimizer = COVID_XRAY_CL.prepare_optimizer(model)

        # Pytorch's Learning Rate Scheduler
        # lr_scheduler_pytorch = CustomReduceLROnPlateau(optimizer=optimizer, patience=3, verbose=True)
        # The step_size=1 parameter means “adjust the LR every time step() is called”. The
        # gamma=0.99 means “multiply the current LR by 0.99 when adjusting the LR”.
        lr_scheduler_pytorch = StepLR(optimizer=optimizer, step_size=1)

        # Wrap it in Avalanche's Plugin:
        lr_scheduler_plugin = LRSchedulerPlugin(lr_scheduler_pytorch)

        # Training Plugins for Strategies
        # Early stopping has never kicked in - disabling and removing from plugins
        # early_stopping_plugin = EarlyStoppingPlugin(patience=3, val_stream_name="test_stream", metric_name="Top1_Acc_Exp")

        # Combine Plugins
        combined_plugins = [lr_scheduler_plugin]

        # STRATEGIES
        model_wrapped_in_strategy = strategy_lambda(model, model_name, optimizer, num_classes, COVID_XRAY_CL.EPOCHS, combined_plugins,
                                                    class_names)

        # # Pass the evaluator to the LR Scheduler Pytorch object
        # lr_scheduler_pytorch.set_evaluator(model_wrapped_in_strategy.evaluator)

        # TRAINING LOOP
        COVID_XRAY_CL.notifier_send(
            'Starting Continual Loop...', print_message=True)

        while True:
            COVID_XRAY_CL.notifier_send("GIVE ME MORE!...", print_message=True)
            # Queue automatically blocks until next item is available, so no need to check.
            scenario_type, current_scenario, submission_id, diagnosis, isLast = COVID_XRAY_CL.INPUT_QUEUE.get()
            # TODO: NEED TO BE ABLE TO TRAIN ON ONE IMAGE AT A TIME
            if scenario_type == "train":
                COVID_XRAY_CL.notifier_send(
                    'Starting Training Scenario...', print_message=True)
                for index, experience in enumerate(current_scenario.train_stream):
                    COVID_XRAY_CL.notifier_send(
                        "Start of experience: " + str(experience.current_experience), print_message=True)
                    COVID_XRAY_CL.notifier_send(
                        "Current Classes: " + str(experience.classes_in_this_experience), print_message=True)

                    COVID_XRAY_CL.notifier_send(
                        str(model_name + " " + strategy_name), print_message=True)

                    # Start Time Nanoseconds
                    train_start_time = time.time_ns()
                    training_results_inc_eval = model_wrapped_in_strategy.train(
                        experiences=experience, eval_streams=[])
                    # End Time Nanoseconds
                    train_time_taken_in_secs = time.time_ns() - train_start_time
                    COVID_XRAY_CL.TRAINING_TEST_RESULTS.append(
                        {'time_taken': train_time_taken_in_secs})

                    # Set to false to prevent evaluations
                    if False:
                        COVID_XRAY_CL.notifier_send('Computing accuracy on the whole test set \n' + model_name + " " + strategy_name,
                                                    print_message=True)
                        evaluation_results = model_wrapped_in_strategy.eval(
                            main_scenario.test_stream)
                        # Write Raw results
                        with open(model_name + "-eval-log.txt", "a") as results_file:
                            results_file.write("\nExperience " + str(index) + "\n\n" +
                                               json.dumps(evaluation_results,
                                                          sort_keys=False,
                                                          indent=4,
                                                          default=lambda o: '<not serializable>')
                                               + "\n" +
                                               "Final Learning Rate: " +
                                               str(lr_scheduler_pytorch.get_last_lr())
                                               + "\n")

                    # Set to true to allow saving a new model checkpoint
                    if False:
                        # Save current model to allow loading later in case of power outage etc. Additionally it could be made to load the best performing model after 5 experiences.
                        COVID_XRAY_CL.save(
                            model=model_wrapped_in_strategy.model, model_name=model_name)

                    if submission_id is not None:
                        # Write results to firebase
                        COVID_XRAY_CL.notifier_send(
                            "Writing training results to Firebase... LEARNT", print_message=True)
                        set_learnt_in_firestore(submission_id)

                COVID_XRAY_CL.notifier_send(
                    'Finished Training Scenario...', print_message=True)
                COVID_XRAY_CL.INPUT_QUEUE.task_done()
                continue
            elif scenario_type == "test":
                # Current Scenario should be one image already transformed through the corresponding transforms
                COVID_XRAY_CL.notifier_send(
                    'Starting Test Scenario...', print_message=True)
                single_image = COVID_XRAY_CL.single_image_loader(
                    test_transform, current_scenario)
                model_wrapped_in_strategy.model.eval()
                model_wrapped_in_strategy.model.to(COVID_XRAY_CL.DEVICE)
                # Inference
                with torch.no_grad():

                    # Start Time Nanoseconds
                    test_start_time = time.time_ns()
                    prediction = model_wrapped_in_strategy.model(single_image)
                    test_time_taken = time.time_ns() - test_start_time

                    # Get the top-1 prediction
                    predictions_output = torch.exp(prediction)
                    final_prediction = torch.argmax(predictions_output)
                    print("PREDICTION:", class_names[final_prediction])
                    print("PERCENTAGE:", round(
                        predictions_output[0][final_prediction].item() * 100, 2))
                    # Print Results
                    COVID_XRAY_CL.notifier_send(
                        "Finished Test Scenario...", print_message=True)

                    COVID_XRAY_CL.INFERENCE_TEST_RESULTS.append(
                        {'time_taken': test_time_taken, 'prediction': class_names[final_prediction]})

                    if submission_id is not None:
                        # Write results to firebase
                        COVID_XRAY_CL.notifier_send(
                            "Writing classification results to Firebase... PREDICTED", print_message=True)

                        probabilities_dict = {}
                        for i in range(len(class_names)):
                            probabilities_dict[class_names[i]
                                               ] = predictions_output[0][i].item()

                        set_prediction_in_firestore(
                            submission_id, class_names[final_prediction], probabilities_dict)

                COVID_XRAY_CL.INPUT_QUEUE.task_done()
                continue
            elif scenario_type == "test_batch":
                # This scenario type is for testing without a submission being made to firebase
                # Current Scenario should be one image already transformed through the corresponding transforms
                COVID_XRAY_CL.notifier_send(
                    'Starting Test Scenario...', print_message=True)
                single_image = COVID_XRAY_CL.single_image_loader(
                    test_transform, current_scenario)
                model_wrapped_in_strategy.model.eval()
                model_wrapped_in_strategy.model.to(COVID_XRAY_CL.DEVICE)
                # Inference
                with torch.no_grad():

                    # Start Time Nanoseconds
                    test_start_time = time.time_ns()
                    prediction = model_wrapped_in_strategy.model(single_image)
                    test_time_taken = time.time_ns() - test_start_time

                    # Get the top-1 prediction
                    predictions_output = torch.exp(prediction)
                    final_prediction = torch.argmax(predictions_output)
                    print("PREDICTION:", class_names[final_prediction])
                    print("PERCENTAGE:", round(
                        predictions_output[0][final_prediction].item() * 100, 2))
                    # Print Results
                    COVID_XRAY_CL.notifier_send(
                        "Finished Test Scenario...", print_message=True)

                    COVID_XRAY_CL.INFERENCE_TEST_RESULTS.append(
                        {'time_taken': test_time_taken, 'prediction': class_names[final_prediction], 'diagnosis': diagnosis, 'image_path': current_scenario})

                    if submission_id is not None:
                        # Write results to firebase
                        COVID_XRAY_CL.notifier_send(
                            "Writing classification results to Firebase... PREDICTED", print_message=True)

                        probabilities_dict = {}
                        for i in range(len(class_names)):
                            probabilities_dict[class_names[i]
                                               ] = predictions_output[0][i].item()

                        set_prediction_in_test_batch_firestore(
                            submission_id, class_names[final_prediction], probabilities_dict, current_scenario, isLast)

                COVID_XRAY_CL.INPUT_QUEUE.task_done()
                continue
            elif scenario_type == "train_time_test":
                COVID_XRAY_CL.notifier_send(
                    'Starting Dummy Training Scenario...', print_message=True)
                for index, experience in enumerate(current_scenario.train_stream):
                    COVID_XRAY_CL.notifier_send(
                        "Start of experience: " + str(experience.current_experience), print_message=True)
                    COVID_XRAY_CL.notifier_send(
                        "Current Classes: " + str(experience.classes_in_this_experience), print_message=True)

                    COVID_XRAY_CL.notifier_send(
                        str(model_name + " " + strategy_name), print_message=True)

                    # Start Time Nanoseconds
                    train_start_time = time.time_ns()
                    training_results_inc_eval = model_wrapped_in_strategy.train(
                        experiences=experience, eval_streams=[])
                    # End Time Nanoseconds
                    train_time_taken_in_secs = time.time_ns() - train_start_time
                    COVID_XRAY_CL.TRAINING_TEST_RESULTS.append(
                        {'time_taken': train_time_taken_in_secs})

                COVID_XRAY_CL.notifier_send(
                    'Finished Training Scenario...', print_message=True)
                COVID_XRAY_CL.INPUT_QUEUE.task_done()
                continue
            elif scenario_type == "test_time_test":
                # This scenario type is for testing without a submission being made to firebase
                COVID_XRAY_CL.notifier_send(
                    'Starting Dummy Test Scenario...', print_message=True)
                single_image = COVID_XRAY_CL.single_image_loader(
                    test_transform, current_scenario)
                model_wrapped_in_strategy.model.eval()
                model_wrapped_in_strategy.model.to(COVID_XRAY_CL.DEVICE)
                # Inference
                with torch.no_grad():

                    # Start Time Nanoseconds
                    test_start_time = time.time_ns()
                    prediction = model_wrapped_in_strategy.model(single_image)
                    test_time_taken = time.time_ns() - test_start_time

                    # Get the top-1 prediction
                    predictions_output = torch.exp(prediction)
                    final_prediction = torch.argmax(predictions_output)
                    print("PREDICTION:", class_names[final_prediction])
                    print("PERCENTAGE:", round(
                        predictions_output[0][final_prediction].item() * 100, 2))
                    # Print Results
                    COVID_XRAY_CL.notifier_send(
                        "Finished Test Scenario...", print_message=True)

                    COVID_XRAY_CL.INFERENCE_TEST_RESULTS.append(
                        {'time_taken': test_time_taken, 'prediction': class_names[final_prediction], 'diagnosis': diagnosis, 'image_path': current_scenario})

                COVID_XRAY_CL.INPUT_QUEUE.task_done()
                continue

    # Save Model Checkpoint
    @staticmethod
    def save_checkpoint(model, optimiser, model_name, training_dataset, model_arch, strategy):
        model.class_to_idx = training_dataset.class_to_idx
        checkpoint = {
            'arch': model_arch,
            'class_to_idx': model.class_to_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'strategy': strategy
        }
        torch.save(checkpoint, model_name + '_checkpoint.pth')

    # Save entire model

    @staticmethod
    def save(model, model_name):
        torch.save(model.state_dict(), model_name + '_model.pth')

    @staticmethod
    def load(model_file_path: str):
        return torch.load(model_file_path, map_location=COVID_XRAY_CL.DEVICE)

    @staticmethod
    def add_test_runs_to_queue():

        # This kind of training is the same as the experiments early with 75 images, except it is 1 image at a time. And there's random inference performed on images while the model is alive.

        random_covid_image_path = "./random_test_image_from_test_1500_set_COVID_19.png"
        random_normal_image_path = "./random_test_image_from_test_1500_set_Normal_97.jpg"
        random_pneumonia_image_path = "./random_test_image_from_test_1500_set_Pneumonia_72.jpg"

        test_image_to_use = random_covid_image_path

        # Add COVID-19 Test to Queue
        COVID_XRAY_CL.add_test_to_queue(test_image_to_use)

        # Single COVID-19 Image Train
        COVID_XRAY_CL.add_train_to_queue(
            image_path="./Datasets/CLCOVIDXRAY-balanced-unsegmented_live_train/train/COVID-19/COVID-19_4121.png",
            label_string="COVID-19")

        # Add COVID-19 Test to Queue
        COVID_XRAY_CL.add_test_to_queue(test_image_to_use)

        # Single Pneumonia Image Train
        COVID_XRAY_CL.add_train_to_queue(
            image_path="./Datasets/CLCOVIDXRAY-balanced-unsegmented_live_train/train/Pneumonia/Pneumonia_4167.png",
            label_string="Pneumonia")

        # Add COVID-19 Test to Queue
        COVID_XRAY_CL.add_test_to_queue(test_image_to_use)

        # Single Normal Image Train
        COVID_XRAY_CL.add_train_to_queue(
            image_path="./Datasets/CLCOVIDXRAY-balanced-unsegmented_live_train/train/Normal/Normal_4101.png",
            label_string="Normal")

        # Add COVID-19 Test to Queue
        COVID_XRAY_CL.add_test_to_queue(test_image_to_use)

    @staticmethod
    def run_time_tests():
        imageSet = []
        # Read covid-xray-test-images.txt into a list
        with open("./covid-xray-test-images.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                lineItems = line.split(" ")
                imagePath = lineItems[0]
                diagnosis = lineItems[1]
                imageUrl = lineItems[2]
                imageSet.append(
                    {'image_path': imagePath, 'diagnosis': diagnosis, 'image_url': imageUrl})

        # Run Test Inference Test First otherwise accuracy will not be calculated properly with training affecting the accuracy
        COVID_XRAY_CL.notifier_send(
            "Starting Test Inference Test...", print_message=True)
        for i in range(COVID_XRAY_CL.TIME_TESTS_IMAGE_LIMIT):
            COVID_XRAY_CL.add_test_to_queue_for_test_time_test(
                image_path=imageSet[i]['image_path'], diagnosis=imageSet[i]['diagnosis'])
        COVID_XRAY_CL.notifier_send(
            "Finished Adding Test Inference Test to queue...", print_message=True)

        # Run Train Inference Test
        COVID_XRAY_CL.notifier_send(
            "Starting Train Inference Test...", print_message=True)
        for i in range(COVID_XRAY_CL.TIME_TESTS_IMAGE_LIMIT):
            COVID_XRAY_CL.add_test_to_queue_for_train_time_test(
                image_path=imageSet[i]['image_path'], label_string=imageSet[i]['diagnosis'])
        COVID_XRAY_CL.notifier_send(
            "Finished Adding Train Inference Test to queue...", print_message=True)

        COVID_XRAY_CL.notifier_send(
            "Once training is complete, dump the report using the report dump endpoint")

    @staticmethod
    def dump_time_test_reports():
        COVID_XRAY_CL.notifier_send(
            "Dumping Train Test Report...", print_message=True)
        # For each test run, dump the results to csv
        with open("./train_test_results.csv", "w") as f:
            f.write("time_taken_ms\n")
            for test_run in COVID_XRAY_CL.TRAINING_TEST_RESULTS:
                f.write(str(test_run['time_taken'] / 1000000) + "\n")

        COVID_XRAY_CL.notifier_send(
            "Dumping Inference Test Report...", print_message=True)
        # For each test run, dump the results to csv
        with open("./inference_test_results.csv", "w") as f:
            f.write("time_taken_ms,prediction,actual,image_path\n")
            for test_run in COVID_XRAY_CL.INFERENCE_TEST_RESULTS:
                f.write(str(test_run['time_taken'] / 1000000) + "," + test_run['prediction'] +
                        "," + test_run['diagnosis'] + "," + test_run['image_path'] + "\n")

    @staticmethod
    def add_train_to_queue(image_path, label_string, submission_id=None):
        if label_string == "COVID-19" or label_string == "COVID19":
            label = 0
        elif label_string == "Normal" or label_string == "NORMAL":
            label = 1
        elif label_string == "Pneumonia" or label_string == "PNEUMONIA":
            label = 2
        else:
            raise Exception("Unknown label string")

        scenario_from_file = tensors_benchmark(
            train_tensors=[([COVID_XRAY_CL.single_image_loader_to_tensor(
                COVID_XRAY_CL.get_train_transform(), image_path)], [label])],
            # all task labels should be zero, task labels are not used for this task incremental
            task_labels=[0],
            # test_tensors=[main_scenario.test_stream[0].dataset],
            test_tensors=[]
        )

        # Add to queue
        COVID_XRAY_CL.INPUT_QUEUE.put(
            ("train", scenario_from_file, submission_id, label_string, None))
        return scenario_from_file

    @staticmethod
    def add_test_to_queue(image_path, submission_id=None):
        COVID_XRAY_CL.INPUT_QUEUE.put(
            ("test", image_path, submission_id, None, None))

    # Test batch goes into a single submission in firebase
    @staticmethod
    def add_test_to_queue_for_testing(image_path, submission_id=None, diagnosis=None, isLast=False):
        COVID_XRAY_CL.INPUT_QUEUE.put(
            ("test_batch", image_path, submission_id, diagnosis, isLast))

    @staticmethod
    def add_test_to_queue_for_train_time_test(image_path, label_string, submission_id=None):
        if label_string == "COVID-19":
            label = 0
        elif label_string == "Normal":
            label = 1
        elif label_string == "Pneumonia":
            label = 2
        else:
            raise Exception("Unknown label string")

        scenario_from_file = tensors_benchmark(
            train_tensors=[([COVID_XRAY_CL.single_image_loader_to_tensor(
                COVID_XRAY_CL.get_train_transform(), image_path)], [label])],
            # all task labels should be zero, task labels are not used for this task incremental
            task_labels=[0],
            # test_tensors=[main_scenario.test_stream[0].dataset],
            test_tensors=[]
        )

        # Add to queue
        COVID_XRAY_CL.INPUT_QUEUE.put(
            ("train_time_test", scenario_from_file, submission_id, label_string, None))
        return scenario_from_file

    @staticmethod
    def add_test_to_queue_for_test_time_test(image_path, submission_id=None, diagnosis=None, isLast=False):
        COVID_XRAY_CL.INPUT_QUEUE.put(
            ("test_time_test", image_path, submission_id, diagnosis, isLast))

    @staticmethod
    def start_live():
        # This live continual learning model begins its life from the transfer learning of the foundation modal
        # a loop is started on which it will feed from a queue and either train or test as per the queue's state.
        # The model has not seen the dataset used in the continual learning experiments

        # num_of_classes = 3

        ####
        # DENSENET161 - Unsegmented - LWF
        ####
        ####

        datasource = DataSource()
        datasource.CLCOVIDXRAY_balanced_unsegmented_live_train()
        base_model, base_model_name = BaseModel.COVIDXRAY_balanced_unsegmented_DENSENET161()
        COVID_XRAY_CL.CURRENT_GLOBAL_MODEL_NAME = "data_" + \
            datasource.candidate_model_name + "_baseModel_" + base_model_name + "_LwF"
        COVID_XRAY_CL().start_cl_engine(datasource.train_data_dir, datasource.test_data_dir, base_model, COVID_XRAY_CL.CURRENT_GLOBAL_MODEL_NAME,
                                        Strategies.LwF, "LwF")


class BaseModel:

    # DENSENET161

    @staticmethod
    def COVIDXRAY_balanced_unsegmented_DENSENET161():
        # Loads PTH foundation model with default 3 classes as pretrained foundation model
        model_file_path = "./Pretrained_Models/5COVIDXRAY_balanced_unsegmented-DENSENET161/model.pth"
        model = COVID_XRAY_CL.load(model_file_path)
        print("Model before changes")
        print(model)
        return model, "COVIDXRAY_balanced_unsegmented_DENSENET161"


class Strategies:

    @staticmethod
    def LwF(model, model_description_name, optimizer, num_classes, epochs_num, plugins, class_names):
        lwf_strategy = LwF(
            model, optimizer, COVID_XRAY_CL.prepare_criterion(),
            train_mb_size=COVID_XRAY_CL.TRAIN_MINIBATCH_SIZE, train_epochs=epochs_num, eval_mb_size=COVID_XRAY_CL.EVAL_MINIBATCH_SIZE,
            alpha=1.0, temperature=2.0,
            evaluator=COVID_XRAY_CL.prepare_evaluator(
                num_classes, model_description_name + "_LwF", class_names),
            plugins=plugins,
            eval_every=COVID_XRAY_CL.EVAL_EVERY,
            device=COVID_XRAY_CL.DEVICE
        )
        return lwf_strategy

def update_queue_from_firebase():
    while True:
        try:
            fb_submissions_to_process = []
            fb_submissions_to_process = get_submissions_from_firestore()
            for submission in fb_submissions_to_process:
                    if not (submission["prediction"] == "UNCONFIRMED") and not (submission["prediction"] == None) and (submission["learntAt"] == None) and submission['imageUrl']:
                        # Add to queue
                        COVID_XRAY_CL.add_train_to_queue(image_path=submission['imageUrl'], label_string=submission['prediction'],
                                                            submission_id=submission["submission_id"])
                    if submission["probabilities"] == None and submission['imageUrl']:
                        # Add to queue
                        COVID_XRAY_CL.add_test_to_queue(submission['imageUrl'], submission["submission_id"])
        except Exception as err:
            print("Error in fetch submissions from firestore loop")
            print(err)
        time.sleep(15)

def update_queue_from_firebase_with_date_checkpoint():
    # current_date should be today - 1 day so the first check always grabs the previous day's submissions too, just in case
    current_date = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=1)
    while True:
        try:
            fb_submissions_to_process = []
            fb_submissions_to_process = get_submissions_from_firestore_with_date(current_date)
            for submission in fb_submissions_to_process:
                    if not (submission["prediction"] == "UNCONFIRMED") and not (submission["prediction"] == None) and (submission["learntAt"] == None) and submission['imageUrl']:
                        # Add to queue
                        COVID_XRAY_CL.add_train_to_queue(image_path=submission['imageUrl'], label_string=submission['prediction'],
                                                            submission_id=submission["submission_id"])
                    if submission["probabilities"] == None and submission['imageUrl']:
                        # Add to queue
                        COVID_XRAY_CL.add_test_to_queue(submission['imageUrl'], submission["submission_id"])
        except Exception as err:
            print("Error in fetch submissions from firestore loop")
            print(err)
        current_date = datetime.datetime.now(pytz.utc)
        time.sleep(15)




if __name__ == '__main__':
    # Start continual Loop here
    covid_cl_loop_thread = threading.Thread(target=COVID_XRAY_CL.start_live)
    covid_cl_loop_thread.start()

    # Start the queue update from firebase loop here
    update_queue_from_firebase_thread = threading.Thread(target=update_queue_from_firebase)
    update_queue_from_firebase_thread.start()

    # # # Start the queue update from firebase loop here
    # update_queue_from_firebase_thread = threading.Thread(target=update_queue_from_firebase_with_date_checkpoint)
    # update_queue_from_firebase_thread.start()


