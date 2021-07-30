try:
    from ...DataLoader import data_loader_creator
    from ...Model import decoderRNN, encoderCNN
except:
    #run from Code
    from DataLoader import data_loader_creator
    from Model import decoderRNN, encoderCNN
import pytest,os
import pandas as pd
dirFile = os.path.dirname(__file__)
# Arrange

trainDir = os.path.join(dirFile, "../../../Data/train-images_old")
valDir = os.path.join(dirFile, "../../../Data/val-images")
assert os.path.exists(trainDir)
assert os.path.exists(valDir)

@pytest.fixture
def data_loader_creator():
    data_loader_creator = data_loader_creator.DataLoaderCreator(trainDir, valDir)
    return data_loader_creator

@pytest.fixture
def load_encoder_and_decoder(data_loader_creator):
    encoder_instance = encoderCNN.EncoderCNN(256)
    decoder_instance = decoderRNN.DecoderRNN(256, 512, len(data_loader_creator.vocab.itos), 1, data_loader_creator.vocab)
    return encoder_instance,decoder_instance


def test_lengths_of_predictions_is_fixed(data_loader_creator,load_encoder_and_decoder):
    encoder_instance, decoder_instance = load_encoder_and_decoder
    # data_loader_creator = load_data_loader
    el = next(iter(data_loader_creator.trn_dl))
    res = encoder_instance(el[0])
    z = decoder_instance.predict(res)
    lengths = [len(el.split()) for el in z]
    assert all([length == decoder_instance.max_seq_length for length in lengths])
