import torch

from nets.yolo import YoloBody
from utils.utils import get_classes

# 将torch的 .pt 转换成 onnx
_defaults = {
    # --------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    # --------------------------------------------------------------------------#
    "model_path": './weights/best_97.pth',
    "classes_path": 'model_data/classes.txt',
    # ---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    "anchors_path": 'model_data/yolo_anchors.txt',
    "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    # ---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    # ---------------------------------------------------------------------#
    "input_shape": [640, 640],
    # ---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    # ---------------------------------------------------------------------#
    "confidence": 0.6,
    # ---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    # ---------------------------------------------------------------------#
    "nms_iou": 0.1,
    # ---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    # ---------------------------------------------------------------------#
    "letterbox_image": True
}
class_names, num_classes = get_classes(_defaults['classes_path'])


def generate():
    # ---------------------------------------------------#
    #   建立yolo模型，载入yolo模型的权重
    # ---------------------------------------------------#
    net = YoloBody(_defaults["anchors_mask"], num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(_defaults["model_path"], map_location=device))
    net = net.fuse().eval()
    print('{} model, and classes loaded.'.format(_defaults["model_path"]))
    return net


model = generate()
model.eval()
batch_size = 1  # 批处理大小
input_shape = (3, 640, 640)  # 输入数据

input_data_shape = torch.randn(batch_size, *input_shape, device="cpu")

torch.onnx.export(model, input_data_shape, "best97.onnx", verbose=True)
