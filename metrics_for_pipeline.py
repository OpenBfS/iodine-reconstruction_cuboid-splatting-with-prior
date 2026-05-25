import torch
import json

import pandas as pd
import numpy as np
from utils_for_pipeline import get_calculated_exposures, get_true_exposures, load_cuboid_state_dict


def calculate_scen_0_rel_sum_cft(training):
    """ Calculates the scenario error across all space-time cells (relevance 0) for the best reconstruction 
    according to loss on training data and validation data and for the starting point of the fine-tuning step."""
    original_scenario = np.load(training.hyperparams["original_scenario_data_directory"])
    original_scenario = torch.from_numpy(original_scenario).to(dtype=torch.float32)
    original_sum = torch.sum(original_scenario)
    pointwise_abs_differences = torch.abs(original_scenario - training.model.best_reconstruction.detach().cpu())
    differences_sum = torch.sum(pointwise_abs_differences)
    scen_0_rel_sum_result = (differences_sum / original_sum) * 100

    validation_pointwise_abs_differences = torch.abs(original_scenario - training.model.validation_best_reconstruction.detach().cpu())
    validation_differences_sum = torch.sum(validation_pointwise_abs_differences)
    validation_scen_0_rel_sum_result = (validation_differences_sum / original_sum) * 100

    reconstructed_prior = training.model.init_reconstruction.detach().cpu()
    pointwise_abs_differences_init = torch.abs(original_scenario - reconstructed_prior)
    differences_sum_init = torch.sum(pointwise_abs_differences_init)
    scen_0_rel_sum_init = (differences_sum_init / original_sum) * 100

    print( f"Der prozentuale Szenariofehler beträgt {scen_0_rel_sum_result} %. -> Rekonstruktion nach Trainingsdaten gewählt")
    print( f"Der prozentuale Szenariofehler beträgt {validation_scen_0_rel_sum_result} %. -> Rekonstruktion nach Validierungsdaten gewählt")
    return scen_0_rel_sum_result, scen_0_rel_sum_init, validation_scen_0_rel_sum_result


def calculate_scen_n_rel_sum(training, n):
    """ Calculates the scenario error across all space-time cells of relevance n (e.g., 1, 5, 10) for the best reconstruction 
    according to loss on training data and validation data and for the starting point of the fine-tuning step. """

    heatmap = np.load(training.hyperparams["heatmap_data_directory"])
    heatmap_mask = heatmap >= n # relevance level
    torch_mask = torch.tensor(heatmap_mask, dtype=torch.bool)

    original_scenario = np.load(training.hyperparams["original_scenario_data_directory"])
    original_scenario = torch.from_numpy(original_scenario).to(dtype=torch.float32)

    pointwise_abs_differences = torch.abs(original_scenario - training.model.best_reconstruction.detach().cpu())
    validation_pointwise_abs_differences = torch.abs(original_scenario - training.model.validation_best_reconstruction.detach().cpu())

    original_scenario_relevant_sum = original_scenario[torch_mask].sum()
    pointwise_abs_differences_relevant_sum = pointwise_abs_differences[torch_mask].sum()
    validation_pointwise_abs_differences_relevant_sum = validation_pointwise_abs_differences[torch_mask].sum()

    scen_n_rel_sum = (pointwise_abs_differences_relevant_sum / original_scenario_relevant_sum) * 100
    validation_scen_n_rel_sum = (validation_pointwise_abs_differences_relevant_sum / original_scenario_relevant_sum) * 100
    print(f"Szenariofehler (Relevanz: {n}): {scen_n_rel_sum}% -> Rekonstruktion nach Trainingsdaten gewählt")
    print(f"Szenariofehler (Relevanz: {n}): {validation_scen_n_rel_sum}% -> Rekonstruktion nach Validierungsdaten gewählt")

    reconstructed_prior = training.model.init_reconstruction.detach().cpu()
    pointwise_abs_differences_init = torch.abs(original_scenario - reconstructed_prior)
    init_pointwise_abs_differences_relevant_sum = pointwise_abs_differences_init[torch_mask].sum()
    init_scen_n_rel_sum = (init_pointwise_abs_differences_relevant_sum / original_scenario_relevant_sum) * 100
    print(f"Szenariofehler bei Initialisierung (Relevanz: {n}): {init_scen_n_rel_sum}%")

    return scen_n_rel_sum, init_scen_n_rel_sum, validation_scen_n_rel_sum


def calculate_prior_scenario_quality(training):
    """ Calculates the prior quality (scenario error of the prior scenario compared to the original scenario).
    For relevance levels 0, 1, 5, 10. """

    original_scenario = np.load(training.hyperparams["original_scenario_data_directory"])
    original_scenario = torch.from_numpy(original_scenario).to(dtype=torch.float32)

    prior_scenario = np.load(training.hyperparams["prior_data_directory"])
    prior_scenario = torch.from_numpy(prior_scenario).to(dtype=torch.float32)

    pointwise_abs_differences = torch.abs(original_scenario - prior_scenario)

    heatmap = np.load(training.hyperparams["heatmap_data_directory"])

    resulting_metrics = []

    for n in [0, 1, 5, 10]:
        heatmap_mask = heatmap >= n # relevance level
        torch_mask = torch.tensor(heatmap_mask, dtype=torch.bool)

        original_scenario_relevant_sum = original_scenario[torch_mask].sum()
        pointwise_abs_differences_relevant_sum = pointwise_abs_differences[torch_mask].sum()

        prior_scen_n_rel_sum = (pointwise_abs_differences_relevant_sum / original_scenario_relevant_sum) * 100
        resulting_metrics.append(prior_scen_n_rel_sum)

    prior_scen_0_rel_sum, prior_scen_1_rel_sum, prior_scen_5_rel_sum, prior_scen_10_rel_sum = tuple(resulting_metrics)

    return prior_scen_0_rel_sum, prior_scen_1_rel_sum, prior_scen_5_rel_sum, prior_scen_10_rel_sum


def calculate_change_during_training(training):
    """ Calculates the change in the Cuboid Reconstruction based on the Cuboid Prior with respect to the change in the reconstructed iodine values of the scenario.
    For relevance levels 0, 1, 5, 10. """

    # starting point of the training
    reconstructed_prior = training.model.init_reconstruction.detach().cpu() #Comparable between runs that use the same pretraining (ID) / cuboid prior
    # result of the training
    reconstructed_scenario = training.model.best_reconstruction.detach().cpu()

    pointwise_abs_differences = torch.abs(reconstructed_prior - reconstructed_scenario)
    
    heatmap = np.load(training.hyperparams["heatmap_data_directory"])

    resulting_metrics = []

    for n in [0, 1, 5, 10]:
        heatmap_mask = heatmap >= n # relevance level
        torch_mask = torch.tensor(heatmap_mask, dtype=torch.bool)

        reconstructed_prior_relevant_sum = reconstructed_prior[torch_mask].sum()
        pointwise_abs_differences_relevant_sum = pointwise_abs_differences[torch_mask].sum()

        change_scen_n_rel_sum = (pointwise_abs_differences_relevant_sum / reconstructed_prior_relevant_sum) * 100
        resulting_metrics.append(change_scen_n_rel_sum)
    
    change_scen_0_rel_sum, change_scen_1_rel_sum, change_scen_5_rel_sum, change_scen_10_rel_sum = tuple(resulting_metrics)

    return change_scen_0_rel_sum, change_scen_1_rel_sum, change_scen_5_rel_sum, change_scen_10_rel_sum


def calculate_train_rel_sum(training, best_model, validation_best_model):
    """ Calculates the path error in percent on the training data.
    For selection of the best model on the training data and validation data. """

    train_paths_data_list = training.load_pathdataset(split="train")
    train_true_exposures = get_true_exposures(train_paths_data_list)
    train_true_exposures_sum = torch.sum(train_true_exposures)

    train_calc_exposures = get_calculated_exposures(train_paths_data_list, best_model)
    train_calc_exposures = train_calc_exposures.detach().cpu()

    validation_train_calc_exposures = get_calculated_exposures(train_paths_data_list, validation_best_model)
    validation_train_calc_exposures = validation_train_calc_exposures.detach().cpu()

    pointwise_abs_diff = torch.abs(train_true_exposures - train_calc_exposures)
    abs_diff_sum = torch.sum(pointwise_abs_diff)

    validation_pointwise_abs_diff = torch.abs(train_true_exposures - validation_train_calc_exposures)
    validation_abs_diff_sum = torch.sum(validation_pointwise_abs_diff)

    train_rel_sum = (abs_diff_sum / train_true_exposures_sum) * 100
    validation_train_rel_sum = (validation_abs_diff_sum / train_true_exposures_sum) * 100
    print(f"Trainingspfadfehler (bestes Modell auf Trainingsdaten): {train_rel_sum}%")
    print(f"Trainingspfadfehler (bestes Modell auf Validierungsdaten): {validation_train_rel_sum}%")
    return train_rel_sum, validation_train_rel_sum


def calculate_test_rel_sum(training, best_model, validation_best_model):
    """ Calculates the path error in percent on the test data.
    For selection of the best model on the training data and validation data. """

    test_paths_data_list = training.load_pathdataset(split="test")
    test_true_exposures = get_true_exposures(test_paths_data_list)
    test_true_exposures_sum = torch.sum(test_true_exposures)

    test_calc_exposures = get_calculated_exposures(test_paths_data_list, best_model)
    test_calc_exposures = test_calc_exposures.detach().cpu()

    validation_test_calc_exposures = get_calculated_exposures(test_paths_data_list, validation_best_model)
    validation_test_calc_exposures = validation_test_calc_exposures.detach().cpu()

    pointwise_abs_diff = torch.abs(test_true_exposures - test_calc_exposures)
    abs_diff_sum = torch.sum(pointwise_abs_diff)

    validation_pointwise_abs_diff = torch.abs(test_true_exposures - validation_test_calc_exposures)
    validation_abs_diff_sum = torch.sum(validation_pointwise_abs_diff)

    test_rel_sum = (abs_diff_sum / test_true_exposures_sum) * 100
    validation_test_rel_sum = (validation_abs_diff_sum / test_true_exposures_sum) * 100
    print(f"Testpfadfehler (bestes Modell auf Trainingsdaten): {test_rel_sum}%")
    print(f"Testpfadfehler (bestes Modell auf Validierungsdaten): {validation_test_rel_sum}%")
    return test_rel_sum, validation_test_rel_sum


def calculate_validation_rel_sum(training, best_model, validation_best_model):
    """ Calculates the path error in percent on the validation data.
    For selection of the best model on the training data and validation data. """

    validation_paths_data_list = training.load_pathdataset(split="validation")
    validation_true_exposures = get_true_exposures(validation_paths_data_list)
    validation_true_exposures_sum = torch.sum(validation_true_exposures)

    validation_calc_exposures = get_calculated_exposures(validation_paths_data_list, best_model)
    validation_calc_exposures = validation_calc_exposures.detach().cpu()

    validation_validation_calc_exposures = get_calculated_exposures(validation_paths_data_list, validation_best_model)
    validation_validation_calc_exposures = validation_validation_calc_exposures.detach().cpu()

    pointwise_abs_diff = torch.abs(validation_true_exposures - validation_calc_exposures)
    abs_diff_sum = torch.sum(pointwise_abs_diff)

    validation_pointwise_abs_diff = torch.abs(validation_true_exposures - validation_validation_calc_exposures)
    validation_abs_diff_sum = torch.sum(validation_pointwise_abs_diff)

    validation_rel_sum = (abs_diff_sum / validation_true_exposures_sum) * 100
    validation_validation_rel_sum = (validation_abs_diff_sum / validation_true_exposures_sum) * 100
    print(f"Validationpfadfehler (bestes Modell auf Trainingsdaten): {validation_rel_sum}%")
    print(f"Validationpfadfehler (bestes Modell auf Validierungsdaten): {validation_validation_rel_sum}%")
    return validation_rel_sum, validation_validation_rel_sum


def calculate_change_in_cuboids(cuboid_state_dict_begin, cuboid_state_dict_end):
    """ Calculates the change in the cuboids during the training process, provided that the adc mechanism was not used.
    Changes in iodine values, shape parameters, distance traveled in space-time, and volume are determined. """

    # Preparations (**2 because the values are stored as raw values)
    for key in ['radiations', 'half_lengths_x', 'half_widths_y', 'half_heights_z']:
        cuboid_state_dict_begin[key] = cuboid_state_dict_begin[key].cpu() ** 2
        cuboid_state_dict_end[key] = cuboid_state_dict_end[key].cpu() ** 2

    change_results = {}
    
    # Iodine values
    differences = (cuboid_state_dict_end['radiations'] - cuboid_state_dict_begin['radiations'])
    avg_change = round((torch.mean(differences).item()), 2)

    num_positive = (differences > 0).sum().item()
    num_negative = (differences < 0).sum().item()
    num_same = (differences == 0).sum().item()
    total_elements = differences.numel()
    positive_share = round((num_positive / total_elements), 2)
    negative_share = round((num_negative / total_elements), 2)
    same_share = round((num_same / total_elements), 2)

    print("radiations -> Avg radiation change:", avg_change)
    print("radiations -> Share of positive/negative/same values:", positive_share, negative_share, same_share)
    change_results['radiations'] = [avg_change, (positive_share, negative_share, same_share)]

    # Positions
    distances = torch.norm(cuboid_state_dict_end['centers'].cpu() - cuboid_state_dict_begin['centers'].cpu(), dim=1) #dim=1 for Euclidian Distance

    min_distance = round((torch.min(distances)).item(), 2)
    mean_distance = round((torch.mean(distances)).item(), 2)
    max_distance = round((torch.max(distances)).item(), 2)

    num_positive = (distances > 0).sum().item()
    num_same = (distances == 0).sum().item()
    total_elements = differences.numel()
    positive_share = round((num_positive / total_elements), 2)
    same_share = round((num_same / total_elements), 2)

    print(f"positions -> Minimum distance: {min_distance}")
    print(f"positions -> Mean distance: {mean_distance}")
    print(f"positions -> Maximum distance: {max_distance}")
    print("positions ->  Share of positive/same values:", positive_share, same_share)

    change_results['centers'] = [mean_distance, (positive_share, same_share)]

    # Length Width Height
    for key in ['half_lengths_x', 'half_widths_y', 'half_heights_z']:

        differences = (cuboid_state_dict_end[key] - cuboid_state_dict_begin[key]) * 2 # *2 because half lengths are saved
        avg_change = round((torch.mean(differences).item()), 2)
        print(key, "-> Avg size change:", avg_change)

        num_positive = (differences > 0).sum().item()
        num_negative = (differences < 0).sum().item()
        num_same = (differences == 0).sum().item()
        total_elements = differences.numel()

        positive_share = round((num_positive / total_elements), 2)
        negative_share = round((num_negative / total_elements), 2)
        same_share = round((num_same / total_elements), 2)
        print(key, "-> Share of positive values:", positive_share)
        print(key, "-> Share of negative values:", negative_share)
        print(key, "-> Share of same values:", same_share)
        
        change_results[key] = [avg_change, (positive_share, negative_share, same_share)]

    # Volume Change overall
    cuboid_volumes_begin = (cuboid_state_dict_begin['half_lengths_x']*2) * (cuboid_state_dict_begin['half_widths_y']*2) * (cuboid_state_dict_begin['half_heights_z']*2)
    cuboid_volumes_end = (cuboid_state_dict_end['half_lengths_x']*2) * (cuboid_state_dict_end['half_widths_y']*2) * (cuboid_state_dict_end['half_heights_z']*2)
    volume_change = round(((torch.sum(cuboid_volumes_end - cuboid_volumes_begin) / torch.sum(cuboid_volumes_begin) * 100).item()), 2)
    print(f"volume overall -> Volume changed by {volume_change} %")

    return change_results['radiations'][0], change_results['radiations'][1], change_results['centers'][0], change_results['centers'][1], change_results['half_lengths_x'][0], change_results['half_lengths_x'][1], change_results['half_widths_y'][0], change_results['half_widths_y'][1], change_results['half_heights_z'][0], change_results['half_heights_z'][1], volume_change
