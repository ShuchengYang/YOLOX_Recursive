import gc

from yolox import *
from needle_dataset import *
import time


# TODO general param setting
version_info = '[V5.0-Recursive-test]'
pre_path = 'E:/proj/'
# pre_path = '/content/drive/MyDrive/'

train_img_folder = pre_path+'needleV2/imgs'
train_ann_folder = pre_path+'needleV2/annotations'
test_img_folder = pre_path+'needleV1/imgs'
test_ann_folder = pre_path+'needleV1/annotations'
model_save_folder = pre_path+'cp'

train_log = pre_path+'cp/'+version_info+'train_log.txt'
train_epo_log = pre_path+'cp/'+version_info+'train_epoch_log.txt'
test_log = pre_path+'cp/'+version_info+'test_log.txt'
test_epo_log = pre_path+'cp/'+version_info+'test_eopch_log.txt'

exp_config_log = pre_path+'cp/'+version_info+'exp_config_log.txt'


train_batch = 1
test_batch = 4

#model number of classes
num_cls = 1
#training
epoch = 100
device = "cpu"
#optimizer
learning_rate = 1e-2
#loss recorder
min_test_loss = 0.6
not_bad_limit = 7.5e-3

with open(exp_config_log, 'w+') as f:
    f.write(f"------ EXPERIMENT {version_info} CONFIGURATION------\n"
            f"train batch : {train_batch}\n"
            f"test batch : {test_batch}\n"
            f"training epoch : {epoch}\n"
            f"device : {device}\n"
            f"learning rate : {learning_rate}\n"
            f"min test loss : {min_test_loss}\n"
            f"not bad limit : {not_bad_limit}\n\n"
            "------ NETWORK CONFIGURATION ------\n"
            "BACKBONE : BASIC DARKNET 53\n"
            "NECK : RECURSIVE\n")


#TODO prepare the dataset
train_data = NeedleData(train_img_folder, train_ann_folder,batch=train_batch, shuffle=True)
test_data = NeedleData(test_img_folder, test_ann_folder, batch=test_batch, shuffle=False)

train_data_size = train_data.get_datalen()
test_data_size = test_data.get_datalen()

#data loader
train_dataloader = train_data.get_dataloader()
test_dataloader = test_data.get_dataloader()


#TODO build the model
#set the training to true so that we got the loss as output
model = Yolox(num_cls=num_cls, training=True).to(device)

#optimizer
opt = torch.optim.Adam(model.parameters(), learning_rate)

#set some variables for training and testing network
#record the time of training
total_train_step = 0
#record the time of testing
total_test_step = 0


#TODO traning stage
model.train()
for i in range(epoch):
    print("")
    train_start =time.time()
    for imgs, targets in train_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        step_start = time.time()
        loss = model(imgs, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()
        step_end = time.time()

        total_train_step = total_train_step + 1
        print(f"e-{i+1} train-prog {round((total_train_step/(train_data_size/train_batch))-i,2)}, loss 【{loss}】")
        #record loss curve
        with open(train_log, 'a+') as f:
            f.write(f"E {i+1} {loss} {total_train_step} {step_end -step_start} | EPOCH loss total_train_step TIME\n")
    train_end=time.time()
    with open(train_epo_log, 'a+') as f:
        f.write(f"T {i + 1} {train_end - train_start} | EPOCH TIME\n")

    # test after each epoch
    model.eval()
    total_test_loss = 0
    #set the training flag to True to get the loss instead
    model.training = True
    with torch.no_grad():
        test_start = time.time()
        for test_imgs, test_targets in test_dataloader:
            test_imgs = test_imgs.to(device)
            test_targets = test_targets.to(device)

            tt_start = time.time()
            test_loss = model(test_imgs, test_targets)
            total_test_loss = total_test_loss + test_loss
            tt_end = time.time()

            total_test_step += 1
            print(f"【test】 e-{i+1} test-prog {round((total_test_step/(test_data_size/test_batch))-i,2)} tloss {test_loss}")
            #record test loss curve
            with open(test_log, 'a+') as f:
                f.write(f"E {i+1} {test_loss} {total_test_step} {tt_end-tt_start} | EPOCH test_loss total_test_loss TIME\n")
        test_end = time.time()
        print(f"ttloss {total_test_loss} avgtloss 【{total_test_loss/test_data_size}】")
        #record ttloss and avgtloss
        with open(test_epo_log, 'a+') as f:
            f.write(f"T {i+1} {total_test_loss} {total_test_loss/test_data_size} {test_end-test_start} | EPOCH total_test_loss AVGTLOSS TIME\n")

    #save model
    model.train()
    print("----------------------------------------------")
    if total_test_loss / test_data_size <= min_test_loss + not_bad_limit:
        print(f"new loss {total_test_loss/test_data_size} <= previous {min_test_loss} + {not_bad_limit} = {min_test_loss+not_bad_limit}")
        print(f"epoch {i+1} train+test finished, save param")
        if total_test_loss / test_data_size <= min_test_loss:
            min_test_loss = total_test_loss / test_data_size
            path = model_save_folder + "/"+version_info+"Ep-" + str(i + 1) + "-NewMin.tar"
        else:
            path = model_save_folder+"/"+version_info+"Ep" + str(i + 1) + "-NotBad.tar"
        torch.save({
            'current i': i,
            'epoch': i + 1,
            'model': model.state_dict(),
        }, path)
        print("param saved")
    else:
        print(f"【skip saving】new avgtloss {total_test_loss / test_data_size} > previous {min_test_loss + not_bad_limit}")
