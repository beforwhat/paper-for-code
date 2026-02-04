def get_model_config(base_dataset_name):
    # 1. 通用模型配置（所有模型共享）
    common_model_config = {
        "backbone": "vgg11",  # 默认Backbone：vgg11，可选：custom_cnn
        "pretrained": False,  # 是否使用预训练模型（仅vgg11支持，CIFAR-10无需预训练）
        "dropout_prob": 0.5,  # Dropout概率，防止过拟合，默认0.5
        "hidden_dim": 512,  # 自定义CNN的隐藏层维度，默认512
    }

    # 2. 按数据集匹配专属模型参数
    dataset_specific_config = {
        "cifar10": {
            "input_size": (3, 32, 32),  # 输入通道数×高度×宽度
            "num_classes": 10,  # 类别数
            "vgg11_config": {
                "num_filters": [64, 128, 256, 512, 512],  # VGG11的卷积层滤波器数量
                "fc_dim": 4096,  # VGG11全连接层维度
            }
        },
        "emnist": {
            "input_size": (1, 28, 28),  # EMNIST为单通道灰度图
            "num_classes": 62,  # EMNIST byclass有62个类别（0-9, a-z, A-Z）
            "vgg11_config": {
                "num_filters": [32, 64, 128, 256, 256],
                "fc_dim": 2048,
            }
        },
        "svhn": {
            "input_size": (3, 32, 32),
            "num_classes": 10,
            "vgg11_config": {
                "num_filters": [64, 128, 256, 512, 512],
                "fc_dim": 4096,
            }
        },
        "femnist": {
            "input_size": (1, 28, 28),
            "num_classes": 62,
            "vgg11_config": {
                "num_filters": [32, 64, 128, 256, 256],
                "fc_dim": 2048,
            }
        }
    }

    # 3. 整合通用配置与数据集专属配置
    model_config = {
        **common_model_config,
        **dataset_specific_config[base_dataset_name],
        "dataset_name": base_dataset_name,
    }

    return model_config