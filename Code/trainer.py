from torch_snippets import *
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from Model import encoderCNN,decoderRNN
from DataLoader import data_loader_creator
from torch import nn

class TrainerAlgo:
    def __init__(self,trn_dl,val_dl,encoder,decoder,optimizer,criterion):
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = criterion



    def  train(self):
        for epoch in range(n_epochs):
            if epoch == 5:
                self.optimizer = torch.optim.AdamW(params, lr=1e-4)
            N = len(self.trn_dl)
            for i, data in enumerate(self.trn_dl):
                trn_loss = self.train_batch(data)
                pos = epoch + (1 + i) / N
                log.record(pos=pos, trn_loss=trn_loss, end='\r')

            N = len(self.val_dl)
            for i, data in enumerate(self.val_dl):
                val_loss = self.validate_batch(data)
                pos = epoch + (1 + i) / N
                log.record(pos=pos, val_loss=val_loss, end='\r')

            log.report_avgs(epoch + 1)

        log.plot_epochs(log=True)
    def train_batch(self,data):
        self.encoder.train() # set the encoder on train Mode
        self.decoder.train() # set the decoder on train Mode

        images, captions, lengths = data

        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths.cpu(),batch_first=True)[0]

        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = self.criterion(outputs, targets)
        self.decoder.zero_grad()
        self.encoder.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    @torch.no_grad()
    def validate_batch(self,data, encoder, decoder, criterion):
        self.encoder.eval()
        self.decoder.eval()
        images, captions, lengths = data

        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths.cpu(), batch_first=True)[0]

        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = self.criterion(outputs, targets)
        return loss


if __name__ == '__main__':
    data_loader_creator_instance = data_loader_creator.DataLoaderCreator()
    trn_dl = data_loader_creator_instance.trn_dl
    val_dl = data_loader_creator_instance.val_dl
    vocab = data_loader_creator_instance.vocab
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = encoderCNN.EncoderCNN(256).to(device)
    decoder = decoderRNN.DecoderRNN(256, 512, len(vocab.itos), 1,vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    n_epochs = 10
    log = Report(n_epochs)

    alg = TrainerAlgo(trn_dl,val_dl,encoder,decoder,optimizer,criterion)
