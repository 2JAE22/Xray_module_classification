import argparse
import importlib
import wandb
import os
import pytorch_lightning as pl
import torch
import timm
import warnings

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy
warnings.filterwarnings("ignore")


def main(config_path, use_wandb=False, sweep_dict=None):
    
    
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    # print("config설정: ",config)


    # 데이터 모듈 동적 임포트 ## .을 기준으로 오른쪽에서 split하여 모듈 경로와 이름을 분리한다.
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    ### 결과: data_module_path= code.dataloader.custom_datamodules.Xray_datamodule, data_module_class = XrayDataModule" 

    #Xray_dataModule(경로) 에서 XrayDataModule이라는 클라스를 가져오는 것.-> 이를 통해 동적으로 임포트 할 수있게 된다.
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )
    ## 즉 여기서 DataModuleClass는custom_datamodules 폴더안-> Xray_datamodule.py안에 있는 XrayDataModule 클래스이다.(CustomDataset은 XrayDataModule 안에 있는 Dataloader역할.)


    # 데이터 모듈 설정
    data_config_path = config.data_config_path ## 값:data_config_path: "configs/data_configs/Xraydata_config.yaml"
    # print("data_config_path: ", data_config_path)

    # augmentation
    augmentation_config_path = config.augmentation_config_path
    # print("augmentation_config_path: ", augmentation_config_path)

    # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    seed = config.get("seed", 42)  
    
    #이 코드로 인해서 XrayDataModule이 train/test dataset를 가져옴.
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    # print("data_module: ", data_module)

    data_module.setup()  # 데이터 모듈에는 setup이라는 메소드가 존재한다. -> setup을 통해서 그 안에 데이터셋 생성 및
    # print("data_module_setup completed")

    # 모델 모듈 동적 임포트
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ## 결과: model_module_path = code.plmodules.smp_module / model_module_class = XrayModelModule

    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )
    ##결과: ModelModuleClass = XrayModelModule

    # 모델 설정
    model = ModelModuleClass(config) #XrayModelModule에  config_modelname,num_clasee,pretrained,들어감.
    # print("model을 print해보겠습니다 얍!: ",model)
    

    # Wandb 로거 설정 (use_wandb 옵션에 따라)
    logger = None
    if use_wandb:
        logger = WandbLogger(project="Xray", name=config.wandb_name)

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )
    # # **Detect Anomaly 설정**
    # torch.autograd.set_detect_anomaly(True)
    
    # 트레이너 설정 -> 이렇게 하면 원래 progressbar 가 나와야 하는데 안 나오는거 보니까 지금 뭔가가 잘못됨..
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        precision='16-mixed',
        strategy=DDPStrategy(find_unused_parameters=True),  # 미사용 파라미터 허용

                
    )
    # print("Start Training!!")

    # 훈련 시작- model은 config.yaml에 적힌것처럼 DeepLabV3PlusModel
    trainer.fit(model, datamodule=data_module)






if __name__ == "__main__":
    import torch, gc
    gc.collect()
    if torch.cuda.is_available(): #CUDA사용 가능 여부 확인 후 캐시 초기화
        torch.cuda.empty_cache()
    

    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    
    #required=True 하면 해당 인자가 반드시 들어가야 함을 의미.
    parser.add_argument(
        "--config", type=str, required=False,default="configs/train_configs/train/config.yaml" ,help="Path to the config file" 
    )

    # action은 특정플래그가 존재하는지 여부체크. 없으면 False,있으면 True 로 store_true가 결정해줌.
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()

    main(args.config, args.use_wandb)
