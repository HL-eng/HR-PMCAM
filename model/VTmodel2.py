import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn import *
from torch.nn.init import uniform_
'''
TextCNN用于对文本特征进行处理  这个地方虽然定义的类是TextCNN，但是已经不是通过TextCNN所提取的特征了
'''
class TextCNN(Module):
    def __init__(self, text_feature_dim=512, hidden_dim=128,  #这个地方的文本特征维度可能需要修改一下
        uniform_value = 0.5):
        super(TextCNN, self).__init__()
        self.uniform_value = uniform_value
        self.hidden_dim = hidden_dim
        self.textual_nn = Sequential(
            Linear(text_feature_dim, 1024),  # 1024->512
            Sigmoid(),
            Linear(1024, self.hidden_dim),  # 1024->512
            Sigmoid(),
        )
        #self.textual_features = textual_features
        # self.textual_nn[0].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        # self.textual_nn[0].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()
        #
        # self.textual_nn[2].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        # self.textual_nn[2].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()

        self.textual_nn[0].apply(lambda module: uniform_(module.weight.data, 0, 0.001))
        self.textual_nn[0].apply(lambda module: uniform_(module.bias.data, 0, 0.001))

        self.textual_nn[2].apply(lambda module: uniform_(module.weight.data, 0, 0.001))
        self.textual_nn[2].apply(lambda module: uniform_(module.bias.data, 0, 0.001))

    #在这个方法当中相当于对文本特征进行下处理，然后进行返回  这个地方的方法不一定正确，需要提前测试
    def forward(self, input):
        return self.textual_nn(input)

class VisualProcess(Module):
    def __init__(self,embedding_weight ,  #此处的用户集合与衣服集合是集合当中的全部
        max_sentence = 83,  text_feature_dim=300,
        visual_feature_dim = 512, hidden_dim=128,
        uniform_value = 0.5):
        super(VisualProcess, self).__init__()
        self.uniform_value = uniform_value
        self.hidden_dim = hidden_dim
        self.visual_nn = Sequential(
            Linear(visual_feature_dim, 1024),  # 2048->512
            Sigmoid(),
            Linear(1024, self.hidden_dim),  # 2048->512
            Sigmoid(),
        )
        # self.visual_nn[0].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        # self.visual_nn[0].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()
        #
        # self.visual_nn[2].apply(lambda module: uniform_(module.weight.data, 0, 0.001)).cuda()
        # self.visual_nn[2].apply(lambda module: uniform_(module.bias.data, 0, 0.001)).cuda()

        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data, 0, 0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data, 0, 0.001))

        self.visual_nn[2].apply(lambda module: uniform_(module.weight.data, 0, 0.001))
        self.visual_nn[2].apply(lambda module: uniform_(module.bias.data, 0, 0.001))

        # self.textcnn = TextCNN(sentence_size=(max_sentence, text_feature_dim), output_size=hidden_dim)
        self.textprocess = TextCNN(text_feature_dim=512, hidden_dim=128, uniform_value = 0.5)

    def forward_u(self, neigh, visual_features, text_features):

        #对视觉特征以及文本特征进行处理
        if not self.visual_nn[0].weight.data.is_cuda:
            neigh_visual_latent = self.visual_nn(cat(                  #下衣负例视觉特征
                [visual_features[n].unsqueeze(0) for n in neigh], 0    # 按行进行拼接
            ))

            neigh_text_latent = self.textprocess(cat(                  #下衣负例视觉特征
                [text_features[n].unsqueeze(0) for n in neigh], 0
            ))

        else :
            neigh_visual_latent = self.visual_nn(cat(  # 下衣负例视觉特征
                [visual_features[n].unsqueeze(0) for n in neigh], 0
            ).cuda())

            neigh_text_latent = self.textprocess(cat(                  #下衣负例视觉特征
                [text_features[n].unsqueeze(0) for n in neigh], 0
            ).cuda())
        return neigh_visual_latent, neigh_text_latent

    #
    def forward_c(self, Is, Js, Ks, visual_features, text_features):
        if not self.visual_nn[0].weight.data.is_cuda:
            # 注意使用unsequeeze方法后，张量的变化形式
            I_visual_latent = self.visual_nn(cat(  # 上衣视觉特征
                [visual_features[str(I)].unsqueeze(0) for I in Is], 0
            ))
            J_visual_latent = self.visual_nn(cat(  # 下衣正例视觉特征
                [visual_features[str(J)].unsqueeze(0) for J in Js], 0
            ))
            K_visual_latent = self.visual_nn(cat(  # 下衣负例视觉特征
                [visual_features[str(K)].unsqueeze(0) for K in Ks], 0
            ))

            #新方法
            I_text_latent = self.textprocess(cat(  # 上衣文本特征
                [text_features[str(I)].unsqueeze(0) for I in Is], 0
            ))
            J_text_latent = self.textprocess(cat(  # 下衣正例文本特征
                [text_features[str(J)].unsqueeze(0) for J in Js], 0
            ))
            K_text_latent = self.textprocess(cat(  # 下衣负例文本特征
                [text_features[str(K)].unsqueeze(0) for K in Ks], 0
            ))

        else:
            with torch.cuda.device(self.visual_nn[0].weight.data.get_device()):
                stream1 = torch.cuda.Stream()
                stream2 = torch.cuda.Stream()
                I_visual_latent = self.visual_nn(cat(
                    [visual_features[str(I)].unsqueeze(0) for I in Is], 0
                ).cuda())
                with torch.cuda.stream(stream1):
                    J_visual_latent = self.visual_nn(cat(
                        [visual_features[str(J)].unsqueeze(0) for J in Js], 0
                    ).cuda())
                with torch.cuda.stream(stream2):
                    K_visual_latent = self.visual_nn(cat(
                        [visual_features[str(K)].unsqueeze(0) for K in Ks], 0
                    ).cuda())

                I_text_latent = self.textprocess(cat(
                    [text_features[str(I)].unsqueeze(0) for I in Is], 0
                ).cuda())
                with torch.cuda.stream(stream1):
                    J_text_latent = self.textprocess(cat(
                        [text_features[str(J)].unsqueeze(0) for J in Js], 0
                    ).cuda())
                with torch.cuda.stream(stream2):
                    K_text_latent = self.textprocess(cat(
                        [text_features[str(K)].unsqueeze(0) for K in Ks], 0
                    ).cuda())


        return I_visual_latent, I_text_latent, J_visual_latent, J_text_latent, K_visual_latent, K_text_latent