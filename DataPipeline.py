from typing import Tuple, List
from tqdm.auto import tqdm
import numpy as np
import torch
import librosa
import datasets
import re
import random

class DataTransform(object):
    '''
    This DataTransform will take care of performing every attack you want to perform on any audio huggingface dataset.
    '''
    def __init__(self, model, attack, device):
    
        self.device = device
        self.model = model  
        self.attack = attack

    def random_transcription_generator(self, data: datasets.dataset_dict.DatasetDict, sentences_for_dict: int = 100, create_sentences: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Generates random transcription in batch with the same vocabulary as that of the dataset provided.
        Args:

        data   : Subset of huggingface dataset for audio
        sentences_for_dict : number of sentences to use for creating dictionary from which words will be chosen randomly for creating random transcription
        create_sentences   : number of example of batch transcription to generate using this method

        Returns
        '''
        #creating our dictionary of words and storing original transcription
        original_transcription = []
        dictionary_of_words = []
        for _, sentence in enumerate(data[:sentences_for_dict]):
            original_transcription.append(sentence.upper())
            for _, word in enumerate(sentence.split(" ")):
                dictionary_of_words.append(word)

        #remove redundant words
        dictionary_of_words = list(set(dictionary_of_words))

        #sorting the list in-place
        dictionary_of_words.sort()

        #removing unneccessary characters from our created dictionary
        removeSymbol = '[\,\?\.\!\-\;\:\"]'
        dictionary_of_words_new = []
        for i in range(len(dictionary_of_words)):
            dictionary_of_words_new.append(re.sub(removeSymbol, '', dictionary_of_words[i]).upper())

        #removing unneccessary characters from our original trancription also
        original_transcription_new = []
        for i in range(len(original_transcription)):
            original_transcription_new.append(re.sub(removeSymbol, '', original_transcription[i]).upper())  

        #generating random transcription whose no. of words is equal to the ground truth
        random_transcription = []
        for _, sentence in enumerate(original_transcription_new[:create_sentences]):
            temp_sentence = ""
            for i in range(len(sentence.split(" "))):
                random_word = random.choice(dictionary_of_words_new)
                if i == len(sentence.split(" "))-1:
                    temp_sentence += random_word
                else:
                    temp_sentence += random_word + " "
            random_transcription.append(temp_sentence)
  
        return np.array(original_transcription_new[:create_sentences]), np.array(random_transcription)


    def apply_attack(self, attack: str, data: datasets.dataset_dict.DatasetDict, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Applies attack on the subset of data provided to it.
        Args:

        attack   : Name of attack to be performed. For example: "FGSM"
        data     : Subset of huggingface dataset for audio
        **kwargs : All the argument that you pass for attack to be performed using ASRAdversarialAttacks class.

        Returns a tuple containing two numpy arrays. One clean audio array and the other perturbed audio array
        '''
        emp1 = []
        emp2 = []
        for i in tqdm(range(len(data))):
            timit_audio = data[i]['array']

            # convert the list of audio signals to a PyTorch tensor
            timit_audio_tensor = torch.from_numpy(timit_audio)
            timit_audio_tensor = timit_audio_tensor.unsqueeze(0)  # add a batch dimension
            emp1.append(timit_audio_tensor)

            if attack == 'FGSM':
                epsilon = kwargs['epsilon']
                if kwargs['targeted'] == True:
                    targeted = kwargs['targeted']
                    assert kwargs['target'] is not None, "Target must not be passed for targeted attacks"
                    target = kwargs['target']
                    temp = self.attack.FGSM_ATTACK(timit_audio_tensor, target[i].replace(" ","|"), epsilon = epsilon, 
                                                   targeted = targeted)
                else:
                    targeted = kwargs['targeted']
                    temp = self.attack.FGSM_ATTACK(timit_audio_tensor, epsilon = epsilon, 
                                                   targeted = targeted)

            elif attack == 'BIM':
                epsilon = kwargs['epsilon']
                alpha = kwargs['alpha']
                num_iter = kwargs['num_iter']
                if kwargs['targeted'] == True:
                    targeted = kwargs['targeted']
                    assert kwargs['target'] is not None, "Target must not be passed for targeted attacks"
                    target = kwargs['target']
                    temp = self.attack.BIM_ATTACK(timit_audio_tensor, target[i].replace(" ","|"), epsilon = epsilon, 
                                                  alpha = alpha, num_iter = num_iter, 
                                                  targeted = targeted)
                else:
                    targeted = kwargs['targeted']
                    temp = self.attack.BIM_ATTACK(timit_audio_tensor, epsilon = epsilon, 
                                                  alpha = alpha, num_iter = num_iter, 
                                                  targeted = targeted)

            elif attack == 'PGD':
                epsilon = kwargs['epsilon']
                alpha = kwargs['alpha']
                num_iter = kwargs['num_iter']
                if kwargs['targeted'] == True:
                    targeted = kwargs['targeted']
                    assert kwargs['target'] is not None, "Target must not be passed for targeted attacks"
                    target = kwargs['target']
                    temp = self.attack.PGD_ATTACK(timit_audio_tensor, target[i].replace(" ","|"), epsilon = epsilon, 
                                                  alpha = alpha, num_iter = num_iter, 
                                                  targeted = targeted)
                else:
                    targeted = kwargs['targeted']
                    temp = self.attack.PGD_ATTACK(timit_audio_tensor, epsilon = epsilon, 
                                                  alpha = alpha, num_iter = num_iter, 
                                                  targeted = targeted)

            elif attack == 'CW':
                epsilon = kwargs['epsilon']
                c = kwargs['c']
                learning_rate = kwargs['learning_rate']
                num_iter = kwargs['num_iter']
                decrease_factor_eps = kwargs['decrease_factor_eps']
                num_iter_decrease_eps = kwargs['num_iter_decrease_eps']
                optimizer = kwargs['optimizer']
                nested = kwargs['nested']
                early_stop = kwargs['early_stop']
                search_eps = kwargs['search_eps']
                if kwargs['targeted'] == True:
                    targeted = kwargs['targeted']
                    assert kwargs['target'] is not None, "Target must not be passed for targeted attacks"
                    target = kwargs['target']
                    temp = self.attack.CW_ATTACK(timit_audio_tensor, target[i].replace(" ","|"), epsilon = epsilon,
                                                c = c, learning_rate = learning_rate, 
                                                num_iter = num_iter, 
                                                decrease_factor_eps = decrease_factor_eps,
                                                num_iter_decrease_eps = num_iter_decrease_eps, 
                                                optimizer = optimizer, nested = nested, 
                                                early_stop = early_stop, search_eps = search_eps, 
                                                targeted = targeted)
                else:
                    targeted = kwargs['targeted']
                    temp = self.attack.CW_ATTACK(timit_audio_tensor, epsilon = epsilon, c = c, 
                                                learning_rate = learning_rate, 
                                                num_iter = num_iter, 
                                                decrease_factor_eps = decrease_factor_eps,
                                                num_iter_decrease_eps = num_iter_decrease_eps, 
                                                optimizer = optimizer, nested = nested, 
                                                early_stop = early_stop, search_eps = search_eps, 
                                                targeted = targeted)

            elif attack == "IMP_CW":
                assert kwargs['target'] is not None, "Target must not be passed for targeted attacks"
                target = kwargs['target']
                epsilon = kwargs['epsilon']
                c = kwargs['c']
                learning_rate1 = kwargs['learning_rate1']
                learning_rate2 = kwargs['learning_rate2']
                num_iter1 = kwargs['num_iter1']
                num_iter2 = kwargs['num_iter2']
                decrease_factor_eps = kwargs['decrease_factor_eps']
                num_iter_decrease_eps = kwargs['num_iter_decrease_eps']
                optimizer1 = kwargs['optimizer1']
                optimizer2 = kwargs['optimizer2']
                nested = kwargs['nested']
                early_stop_cw = kwargs['early_stop_cw']
                search_eps_cw = kwargs['search_eps_cw']
                alpha = kwargs['alpha']
                temp = self.attack.IMPERCEPTIBLE_ATTACK(torch.nn.functional.pad(timit_audio_tensor, (0, 1000)), 
                                                        target[i].replace(" ","|"), epsilon = epsilon, c = c, 
                                                        learning_rate1 = learning_rate1, 
                                                        learning_rate2 = learning_rate2, 
                                                        num_iter1 = num_iter1, 
                                                        num_iter2 = num_iter2, 
                                                        decrease_factor_eps = decrease_factor_eps, 
                                                        num_iter_decrease_eps = num_iter_decrease_eps, 
                                                        optimizer1 = optimizer1, 
                                                        optimizer2 = optimizer2, 
                                                        nested = nested, 
                                                        early_stop_cw = early_stop_cw, 
                                                        search_eps_cw = search_eps_cw, 
                                                        alpha = alpha)

            emp2.append(torch.from_numpy(temp))
        return np.array(emp1), np.array(emp2)
  
    def pad_audios(self, audios: np.ndarray) -> np.ndarray:
        '''
        Pads audios to the longest sequence in the batch passed to this method.
        Args:
        
        audios: Audios in batch.
        
        Returns padded audios
        '''
        max_length = np.max([x.shape[1] for x in tqdm(audios, total=len(audios))])
        padded_audios = np.array([torch.nn.functional.pad(x, (0, max_length-x.shape[1])) for x in tqdm(audios, total = len(audios))])
        return padded_audios
    
    def stft_computer(self, audios, n_fft = 512, hop_length = 160, win_length = 480, window = torch.hann_window(480), return_db: bool= True) -> List:
        '''
        Computes STFT on the audios in batch.
        Args:
        
        audios : Audios in batch
        n_fft  : number of fft points
        hop_length : hop length argument of STFT
        win_length : window length of the windowing function of STFT (Should be less than the n_fft)
        window     : Pytorch window function to be passed here with window size equal to the win_length
        return_db  : If the user want dB transformed magnitude part of STFT then True else False.
        
        Returns list in which STFT of audios is stored
        '''
        n_fft = n_fft 
        hop_length = hop_length
        win_length = win_length
        window = window

        stft_abs_list = []
        stft_abs_db_list = []

        # loop through the audio files in the train subset
        for i in tqdm(range(len(audios))):

            stft_clean = torch.stft(audios[i][0], n_fft=n_fft, 
                                    hop_length=hop_length, win_length=win_length, 
                                    window=window, center=False, normalized=False, 
                                    return_complex=True).numpy()

            stft_abs = np.abs(stft_clean)
            stft_abs_db = librosa.amplitude_to_db(stft_abs)

            stft_abs_list.append(stft_abs)
            stft_abs_db_list.append(stft_abs_db)

        if return_db:
            return stft_abs_db_list

        else:
            return stft_abs_list
    
    def zero_crossing_computer(self, audios) -> List:
        '''
        Calculates zero crossing of the audios passed in batch to this method.
        Args:
        
        audios: Audios in batch.
        
        Returns: List of integers representing zero crossing value of audios which was passed in batch as input to this method.
        '''
        zc = []
        for i in range(len(audios)):
            audio1 = audios[i]
            zcr1 = librosa.zero_crossings(audio1).sum()
            zc.append(zcr1)
        return zc
  
    def feature_combiner(self, zc_clean, stft_clean, zc_pert, stft_pert) -> Tuple[List, List]:
        '''
        This method combines two features (zero_crossing and STFT) into a horizontal flattened array.
        Args:
        
        zc_clean   : zero crossing values of clean audios
        stft_clean : STFTs of clean audios
        zc_pert    : zero crossing values of perturbed audios
        stft_pert  : STFTs of perturbed audios
        
        Returns tuple containing two lists. One is the X-features and other the Y-labels.
        '''
        combineFeatures = []
        list_stft = stft_clean 
        list_zc = zc_clean
        y_label = []

        for stft, zc in tqdm(zip(list_stft, list_zc), total=len(list_zc)): # for clean dataset
            flattened_stft = stft.flatten()
            combinedFeature = np.hstack((flattened_stft, zc))
            combineFeatures.append(combinedFeature)
            y_label.append(1)

        list_stft = stft_pert
        list_zc = zc_pert

        for stft, zc in tqdm(zip(list_stft, list_zc), total=len(list_zc)): # for perturbed dataset
            flattened_stft = stft.flatten()
            combinedFeature = np.hstack((flattened_stft, zc))
            combineFeatures.append(combinedFeature)
            y_label.append(0)

        return combineFeatures, y_label