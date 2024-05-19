import time
import warnings, datetime
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from torch.nn import functional as F
from torchinfo import summary
from util.data import load_dataset
from util.base import BaseSynthesizer
from util.transformer import BGMTransformer
from util.model_test import fix_random_seed
from datetime import datetime
import logging
class iter_schedular:
    def __init__(self):
        self.G_iter = 0
        self.G_like_iter = 0
        self.G_liker_iter = 0
        self.D_iter = 0
        self.oD_iter = 0
        self.ae_iter = 0
        self.ae_g_iter = 0

def term_check(term, iter):
    if term == 0:
        return False
    return (iter % term) == (term - 1)

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_grad): 
    alpha = torch.rand(real_data.size(0), 1, 1, device=device) 
    alpha = alpha.repeat(1, 1, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_grad 
    return gradient_penalty

class AEGANSynthesizer(BaseSynthesizer):
    
    def __init__(self, config):
        
        self.G_model = config['G_model']; self.embedding_dim = config['embedding_dim']; self.G_args = config['G_args']
        self.G_lr = config['G_lr']; self.G_beta = config['G_beta']; self.G_l2scale = config['G_l2scale']
        self.G_l1scale = config['G_l1scale']; self.G_learning_term = config['G_learning_term']
        self.likelihood_coef = config['likelihood_coef']; self.likelihood_learn_start_score = config['likelihood_learn_start_score']
        self.likelihood_learn_term = config['likelihood_learn_term']; self.kinetic_learn_every_G_learn = config['kinetic_learn_every_G_learn']
        
        self.D_model = config['D_model']; self.dis_dim = config['dis_dim']; self.lambda_grad = config['lambda_grad']
        self.D_lr = config['D_lr']; self.D_beta = config['D_beta']; self.D_l2scale = config['D_l2scale']; self.D_learning_term = config['D_learning_term']
        self.D_leaky = config['D_leaky']; self.D_dropout = config['D_dropout'];
        
        self.En_model = config['En_model']; self.compress_dims = config['compress_dims']; self.AE_lr = config['AE_lr']
        self.AE_beta = config['AE_beta']; self.AE_l2scale = config['AE_l2scale']; self.loss_factor = config['loss_factor']
        self.ae_learning_term = config['ae_learning_term']; self.ae_learning_term_g = config['ae_learning_term_g']
        self.De_model = config['De_model']; self.decompress_dims = config['decompress_dims']; self.L_func = config['L_func']
        
        self.rtol = config['rtol']; self.atol = config['atol']; self.batch_size = config['batch_size']; self.epochs = config['epochs']
        self.random_num = config['random_num']; self.GPU_NUM = config['GPU_NUM']; self.save_loc = config['save_loc']
        self.GPU_NUM = config["GPU_NUM"]; self.data_name = config['data_name']
        
        #self.writer = config["writer"]
        self.device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu') 

        
    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        #self.train = train_data.copy()
        logging.info("Fitting the Transformer")
        self.transformer = BGMTransformer(meta = meta_data, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        logging.info("Transformer fitted")
        train_data = self.transformer.transform(train_data)
        logging.info("Data transformed")
        self.test = test_data
        self.meta = meta_data
        
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=48)
        data_dim = self.transformer.output_dim
        logging.info("Data loading finished")

        encoder = self.En_model(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = self.De_model(self.embedding_dim, self.decompress_dims, data_dim).to(self.device)
        self.generator = self.G_model(self.G_args, self.embedding_dim).to(self.device)
        optimizerEn = optim.Adam(encoder.parameters(), lr = self.AE_lr, betas= self.AE_beta, weight_decay= self.AE_l2scale)
        optimizerDe = optim.Adam(self.decoder.parameters(), lr = self.AE_lr, betas= self.AE_beta, weight_decay= self.AE_l2scale)
        
        optimizerG = optim.Adam(self.generator.parameters(), lr = self.G_lr, betas= self.G_beta, weight_decay = self.G_l2scale)

        if self.D_learning_term != 0:
            self.discriminator = self.D_model(self.embedding_dim, self.dis_dim, self.D_leaky, self.D_dropout).to(self.device)
            optimizerD = optim.Adam(self.discriminator.parameters(), lr = self.D_lr, betas= self.D_beta, weight_decay = self.D_l2scale)
                        
        logging.info("GAN and optimizer and AE Ready")
        iter = 0
        iter_s = iter_schedular()
        track_score_dict, save_score_dict = {}, {}
        logging.info(f"Encoder: {summary(encoder)}")
        logging.info(f"Decoder: {summary(self.decoder)}")
        logging.info(f"Generator: {summary(self.generator)}")
        logging.info(f"Discriminator: {summary(self.discriminator)}")
        mean_z = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std_z = mean_z + 1
        train_start = time.time()
        logging.info("Training Now")
        print("epochs = ", self.epochs) 
        for i in range(self.epochs):
            epoch_start = time.time()
            print("Current epoch = ", i)
            logging.info(f"Current epoch = {i}, starting at {epoch_start} , {datetime.now().strftime('%H:%M:%S.%f')}")
            for _, data in enumerate(loader): 
                iter += 1

                ######## Real data Generation #########
                real = data[0].to(self.device)

                ######## AutoEncoder Learning AE loss #########
                if term_check(self.ae_learning_term, iter):
                    #logging.info(f"Training AE loss at iter {iter_s.ae_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    emb = encoder(real) 
                    rec, sigmas = self.decoder(emb)
                    loss_1, loss_2 = self.L_func( 
                        rec, real, sigmas, emb, self.transformer.output_info, self.loss_factor)
                    loss_ae = loss_1 + loss_2
                    optimizerEn.zero_grad()
                    optimizerDe.zero_grad()
                    loss_ae.backward()
                    optimizerEn.step()
                    optimizerDe.step()
                    self.decoder.sigma.data.clamp_(0.01, 1.0)
                    ##self.writer.add_scalar('losses/AE_loss', loss_ae, iter_s.ae_iter)
                    iter_s.ae_iter += 1
                    #logging.info(f"Finished training AE loss at iter {iter_s.ae_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    #print("AE_loss = ", loss_ae, "iter = ", iter_s.ae_iter)
                ######## AutoEncoder Learning Gan loss #########
                if term_check(self.ae_learning_term_g, iter):
                    logging.info(f"Training AE_G1 and AE_G2 loss at iter {iter_s.ae_g_iter} at time {datetime.now().strftime('%H:%M:%S.%f')}")
                    fakez = torch.normal(mean=mean_z, std=std_z)
                    logging.info(f"Fakez generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    fake_h = self.generator(fakez)
                    logging.info(f"Fake_h generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    rec_syn, _ = self.decoder(fake_h)
                    logging.info(f"rec_syn generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    emb_syn = encoder(rec_syn) 
                    logging.info(f"emb_syn generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    loss_ae_g1 = ((emb_syn - fake_h) ** 2).mean()
                    logging.info(f"loss_ae_g1 generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerEn.zero_grad()
                    logging.info(f"optimizerEn zero_grad {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerDe.zero_grad()
                    logging.info(f"optimizerDe zero_grad {datetime.now().strftime('%H:%M:%S.%f')}")
                    loss_ae_g1.backward()
                    logging.info(f"loss_ae_g1 backward {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerEn.step()
                    logging.info(f"optimizerEn step {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerDe.step()
                    logging.info(f"optimizerDe step {datetime.now().strftime('%H:%M:%S.%f')}")
                    self.decoder.sigma.data.clamp_(0.01, 1.0)
                    logging.info(f"clamped sigma {datetime.now().strftime('%H:%M:%S.%f')}")
                    real_h = encoder(real)
                    logging.info(f"real_h generated {datetime.now().strftime('%H:%M:%S.%f')}") 
                    loss_ae_g2 = -torch.mean(self.discriminator(real_h))
                    logging.info(f"loss_ae_g2 generated {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerEn.zero_grad()
                    logging.info(f"optimizerEn zero_grad {datetime.now().strftime('%H:%M:%S.%f')}")
                    loss_ae_g2.backward()
                    logging.info(f"loss_ae_g2 backward {datetime.now().strftime('%H:%M:%S.%f')}")
                    optimizerEn.step()
                    logging.info(f"optimizerEn step {datetime.now().strftime('%H:%M:%S.%f')}")
                    iter_s.ae_g_iter += 1
                    logging.info(f"Finished training AE_G1 and AE_G2 loss at iter {iter_s.ae_g_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                ######## update inner discriminator #########
                if term_check(self.D_learning_term, iter):
                    #logging.info(f"Training D1 loss at iter {iter_s.D_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    real_h = encoder(real)       
                    fakez = torch.normal(mean=mean_z, std=std_z)
                    fake_h = self.generator(fakez)
                    y_fake = self.discriminator(fake_h)
                    y_real = self.discriminator(real_h)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    pen = calc_gradient_penalty(self.discriminator, real_h, fake_h, self.device, self.lambda_grad)                
                    loss_d = loss_d + pen 
                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()
                    ##self.writer.add_scalar('losses/D1_loss', loss_d, iter_s.D_iter)
                    iter_s.D_iter += 1
                    #logging.info(f"Finished training D1 loss at iter {iter_s.D_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    #print("D1_loss = ", loss_d, "iter = ", iter_s.D_iter)
                ######### update generator with inner discri W-GAN Loss ##########
                if self.D_learning_term != 0 and term_check(self.G_learning_term, iter):
                    #logging.info(f"Training G1 loss at iter {iter_s.G_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    fakez = torch.normal(mean=mean_z, std=std_z)
                    fake_h = self.generator(fakez)
                    y_fake = self.discriminator(fake_h)
                    loss_g = -torch.mean(y_fake)
                    reg_g = 0
                    if self.G_l1scale is not None and self.G_l1scale != 0:
                        reg_g = self.G_l1scale * sum([i.abs().sum() for i in self.generator.parameters()])
                    optimizerG.zero_grad()
                    (loss_g + reg_g).backward()
                    optimizerG.step()
                    #logging.info("losses/G1_loss", loss_g, iter_s.G_iter)
                    ##self.writer.add_scalar('losses/G1_loss', loss_g, iter_s.G_iter)
                    iter_s.G_iter += 1
                    #logging.info(f"Finished training G1 loss at iter {iter_s.G_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    #print("G1_loss = ", loss_g, "iter = ", iter_s.G_iter)
                    if self.kinetic_learn_every_G_learn:
                        real_h = encoder(real) # 수정
                        likelihood_loss, likelihood_reg_loss = self.generator.compute_likelihood_loss(real_h)
                        if (likelihood_reg_loss is not None):
                            optimizerG.zero_grad()
                            likelihood_reg_loss.backward()
                            optimizerG.step()
                            ##self.writer.add_scalar('losses/likelihood_reg_loss', likelihood_reg_loss, iter_s.G_liker_iter)
                            iter_s.G_liker_iter += 1
                            #print("likelihood_reg_loss = ", likelihood_reg_loss, "iter = ", iter_s.G_liker_iter)
                        #logging.info(f"Kinetic learning G1 finished at iter {iter_s.G_liker_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                ######## update generator with Likelihood Loss ##########
                if term_check(self.likelihood_learn_term, iter):
                    #logging.info(f"Training likelihood loss at iter {iter_s.G_like_iter} at time {datetime.now().strftime('%H:%M:%S')}")
                    real_h = encoder(real)
                    likelihood_loss, likelihood_reg_loss = self.generator.compute_likelihood_loss(real_h)

                    likelihood_cal = True
                    if (likelihood_reg_loss is not None and self.likelihood_coef != 0):
                        last_like_loss = likelihood_loss * self.likelihood_coef + likelihood_reg_loss
                    elif (likelihood_reg_loss is not None):
                        last_like_loss = likelihood_reg_loss
                    elif (self.likelihood_coef != 0):
                        last_like_loss = likelihood_loss * self.likelihood_coef
                    else:
                        likelihood_cal = False
                    
                    if (likelihood_cal == True):
                        reg_g = 0
                        if self.G_l1scale is not None and self.G_l1scale != 0:
                            reg_g += self.G_l1scale * sum([i.abs().sum() for i in self.generator.parameters()])
                        optimizerG.zero_grad()
                        (last_like_loss + reg_g).backward()
                        optimizerG.step()

                    ##self.writer.add_scalar('losses/likelihood_loss', likelihood_loss, iter_s.G_like_iter)
                    iter_s.G_like_iter += 1
                    #print("likelihood_loss = ", likelihood_loss, "iter = ", iter_s.G_like_iter)
                    if (likelihood_reg_loss is not None):
                        ##self.writer.add_scalar('losses/likelihood_reg_loss', likelihood_reg_loss, iter_s.G_liker_iter)
                        iter_s.G_liker_iter += 1
                       #print("likelihood_reg_loss = ", likelihood_reg_loss, "iter = ", iter_s.G_liker_iter)     
                    #logging.info(f"Finished training likelihood loss at iter {iter_s.G_like_iter} at time {datetime.now().strftime('%H:%M:%S')}")
            logging.info(f"Saving losses to wandb of finished epoch {i}. Took {time.time() - epoch_start} seconds")
            # self.writer.add_scalar("AE_loss", loss_ae, i)
            # self.writer.add_scalar("AE_G1_loss", loss_ae_g1, i)
            # self.writer.add_scalar("AE_G2_loss", loss_ae_g2, i)
            # self.writer.add_scalar("D1_loss", loss_d, i)
            # self.writer.add_scalar("G1_loss", loss_g, i)
            # self.writer.add_scalar("likelihood_loss", likelihood_loss, i)
            # self.writer.add_scalar("likelihood_reg_loss", likelihood_reg_loss, i)
            # self.writer.add_scalar("epoch", i)
            loss_ae = loss_ae.detach().cpu().item()
            loss_ae_g1 = loss_ae_g1.detach().cpu().item()
            loss_ae_g2 = loss_ae_g2.detach().cpu().item()
            loss_d = loss_d.detach().cpu().item()
            loss_g = loss_g.detach().cpu().item()
            likelihood_loss = likelihood_loss.detach().cpu().item()
            likelihood_reg_loss = likelihood_reg_loss.detach().cpu().item()
            wandb.log({"AE_loss": loss_ae, "iter_s.ae_iter": iter_s.ae_iter, 
                        "AE_G1_loss": loss_ae_g1, "AE_G2_loss": loss_ae_g2,  "iter_s.ae_g_iter": iter_s.ae_g_iter,
                        "D1_loss": loss_d, "iter_s.D_iter": iter_s.D_iter,
                        "G1_loss": loss_g, "iter_s.G_iter": iter_s.G_iter, 
                        "likelihood_loss": likelihood_loss, "iter_s.G_like_iter": iter_s.G_like_iter,
                        "likelihood_reg_loss": likelihood_reg_loss, "iter_s.G_liker_iter": iter_s.G_liker_iter, "epoch": i}) 
            print("AE_loss = ", loss_ae, "iter = ", iter_s.ae_iter)
            print("AE_G1_loss = ", loss_ae_g1, "iter = ", iter_s.ae_g_iter)
            print("AE_G2_loss = ", loss_ae_g2, "iter = ", iter_s.ae_g_iter)
            print("D1_loss = ", loss_d, "iter = ", iter_s.D_iter)
            print("G1_loss = ", loss_g, "iter = ", iter_s.G_iter)
            print("likelihood_loss = ", likelihood_loss, "iter = ", iter_s.G_like_iter)
            print("likelihood_reg_loss = ", likelihood_reg_loss, "iter = ", iter_s.G_liker_iter)
        
        print("Training time: ", time.time() - train_start)
        wandb.log({"Training time":(time.time() - train_start)})
        #self.writer.add_scalar("Training Time", (time.time() - train_start))
        print("Training finished")
        #return the total loss
        return loss_ae_g1
    
    def sample(self, n, z_vector = False):
        self.generator.eval()
        self.decoder.eval()

        mean_z = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std_z = mean_z + 1

        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            fakezs = torch.normal(mean=mean_z, std=std_z)
            fake = self.generator(fakezs)
            fake, _  = self.decoder(fake)
            
            fake = apply_activate(fake, self.transformer.output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        self.generator.train()
        self.decoder.train()
        result = self.transformer.inverse_transform(data, None)
        if z_vector:
            return result, fakezs
        else:
            return result

    def model_load(self, checkpoint, choosed_model):
        dataset_name = checkpoint["arg"]["data_name"]
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name, benchmark=True)
        
        self.train = train_data.copy()
        self.transformer = BGMTransformer(meta_data, random_seed=self.random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        
        self.test = test_data
        self.meta = meta_data
        
        data_dim = self.transformer.output_dim
        
        self.decoder = self.De_model(self.embedding_dim, self.decompress_dims, data_dim).to(self.device)
        self.generator = self.G_model(self.G_args, self.embedding_dim).to(self.device)
        if "name" in checkpoint:
            self.generator.load_state_dict(checkpoint["model"][choosed_model]['generator'])
            self.decoder.load_state_dict(checkpoint["model"][choosed_model]['decoder'])
            self.generator.eval()
            self.decoder.eval()
        else:
            self.generator.load_state_dict(checkpoint["info"][choosed_model]['generator'])
            self.decoder.load_state_dict(checkpoint["info"][choosed_model]['decoder'])
            self.generator.eval()
            self.decoder.eval()
