import torch
import numpy as np

class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'model.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'model.pth')
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{"Anomaly Detection":<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model:":<20}{"LLMDetector":<20}{"Seed:":<20}{args.seed:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    # print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    # print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    # if args.task_name in ['long_term_forecast', 'short_term_forecast']:
    #     print("\033[1m" + "Forecasting Task" + "\033[0m")
    #     print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
    #     print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
    #     print(f'  {"Inverse:":<20}{args.inverse:<20}')
    #     print()
    #
    # if args.task_name == 'imputation':
    #     print("\033[1m" + "Imputation Task" + "\033[0m")
    #     print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
    #     print()

    # if args.task_name == 'anomaly_detection':
    #     print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
    #     print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
    #     print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Gnn:":<20}{args.gnn:<20}{"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
    # print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Patch_len:":<20}{args.patch_len:<20}{"Stride:":<20}{args.stride:<20}')
    # print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"d model:":<20}{args.d_model:<20}')
    # print(f'  {"C Out:":<20}{args.c_out:<20}{"e layers:":<20}{args.e_layers:<20}')
    # print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    # print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    # print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    # print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.lr:<20}')
    # print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    # print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    # print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    # p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    # print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print()
