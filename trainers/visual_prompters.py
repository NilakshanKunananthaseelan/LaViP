"""
    source1(VP): https://github.com/hjbahng/visual_prompting
"""
import torch
import torch.nn as nn
from torchvision import transforms


# inverting RGB transform of CLIP
inv_normalize = transforms.Normalize(
                                    mean=[-0.48145466/0.26862954, 
                                          -0.4578275/0.26130258, 
                                          -0.40821073/0.27577711],
                                    std=[1/0.26862954, 
                                         1/0.26130258, 
                                         1/0.27577711]
                                    )
embed_dim_dict = {'RN50':1024,
                    'RN101': 512,
                    'ViT-B/32': 512,
                    
                    'RN50x16':768,
                    'ViT-B/16': 512,
                    'ViT-L/14': 768,

        }

ab_dict = {224: {
            
                            20 : (240, 204),
                            24 : (240, 240),
                            28 : (294, 224),
                            32 : (288, 256),
                            36 : (288, 282),
                            40 : (320, 276),
                            44 : (320, 297),
                            48 : (352, 288),
                            52 : (344, 312),
                            56 : (336, 336),
                            60 : (360, 328)  },
                    288: {
                            20 : (268, 240),
                            24 : (288, 264),
                            28 : (312, 280),
                            32 : (384, 256),
                            36 : (336, 324),
                            40 : (372, 320),
                            44 : (366, 352),
                            48 : (384, 360),
                            52 : (416, 354),
                            56 : (406, 384),
                            60 : (432, 380),
                            
                    
                                       },
                    336:{
                        20 : (316, 240),
                        24 : (312, 288),
                        28 : (336, 308),
                        32 : (384, 304),
                        36 : (360, 360),
                        40 : (384, 370),
                        44 : (438, 352),
                        48 : (432, 384),
                        52 : (426, 416),
                        56 : (448, 420),
                        60 : (460, 432)},

                    384: {
                            20 : (312, 280),
                            24 : (324, 320),
                            28 : (356, 336),
                            32 : (384, 352),
                            36 : (432, 348),
                            40 : (430, 384),
                            44 : (440, 408),
                            48 : (448, 432),
                            52 : (498, 416),
                            56 : (492, 448),
                            60 : (486, 480),
                            },
                    448:{
                            20 : (321, 320),
                            24 : (384, 318),
                            28 : (392, 360),
                            32 : (416, 384),
                            36 : (432, 412),
                            40 : (480, 408),
                            44 : (528, 404),
                            48 : (480, 480),
                            52 : (528, 468),
                            56 : (588, 448),
                            60 : (576, 485)}
                    

                            }

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        self.args = args
        assert args.TRAINER.NAME in ['VPWB'], 'trainer name must be one of VPWB, MMPROMPT,MMPROMPT_COND'
        trainer_config = {
            'VPWB': (args.TRAINER.VPWB.PROMPT_SIZE, args.TRAINER.VPWB.IMAGE_SIZE),
        }

        pad_size, image_size = trainer_config[args.TRAINER.NAME]
        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        n_samples = x.shape[0]
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        x_prompted = x + prompt
        return x_prompted


class PadPrompterLaViP(nn.Module):
    "Train text prompt initialized matrix and gloabl prompt matrix"
    def __init__(self, args,**kwargs):
        super(PadPrompterLaViP, self).__init__()
        self.args = args
       
        assert args.TRAINER.NAME in ['VPWB','LAVIP'], 'trainer name must be one of VPWB,LAVIP'
        trainer_config = {
            'VPWB': (args.TRAINER.VPWB.PROMPT_SIZE, args.TRAINER.VPWB.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
            'LAVIP': (args.TRAINER.LAVIP.PROMPT_SIZE, args.TRAINER.LAVIP.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
          
        }

        self.pad_size, self.image_size,self.backbone = trainer_config[args.TRAINER.NAME]
        mat_dict = ab_dict[self.image_size]
        vlm_dim = embed_dim_dict[self.backbone]
        n_classes = kwargs['n']

        
        if n_classes >15:
            m = n_classes
            r = 32
            
        else:
            m = 64
            r = 16

 
        a,b = mat_dict[self.pad_size]
        self.base_size = self.image_size - self.pad_size*2
        
        


        self.B1 = nn.Parameter(torch.randn([1,b, r]),requires_grad=True)
        self.B2 = nn.Parameter(torch.randn([1,r, m]),requires_grad=True)

        self.M = nn.Parameter(torch.randn([m, n_classes]),requires_grad=True)
        


        self.fc1 = nn.Linear(vlm_dim, a,bias=False)
    
        self.scaleB = nn.Linear(vlm_dim, m,bias=False)
        self.shiftB = nn.Linear(vlm_dim, m,bias=False)

       
       
       
 

    def load_pad_vectors(self,A, B, device='cpu'):
        
        matrix = torch.bmm(A, B)
        sizes = lambda c, h, w, p: [c * p * h, c * p * h, c * (h - 2 * p) * p, c * (h - 2 * p) * p]
        submatrix_sizes = sizes(3, self.image_size, self.image_size, self.pad_size)
        
        # Unravel the matrix and split it into four parts
        submatrices = torch.split(matrix.view(matrix.size(0),-1), submatrix_sizes, dim=1)

        # Reshape and move the submatrices to the specified device
        self.pad_up = submatrices[0].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_down = submatrices[1].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_left = submatrices[2].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)
        self.pad_right = submatrices[3].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)

        



    def forward(self, x,embedding,image_embedding=None):
        
        embedding = embedding.to(x.device)
        
        image_embedding = image_embedding.to(x.device)
        batch_size = image_embedding.size(0)
        self.A = self.fc1(embedding)
        self.eA = torch.matmul(self.M,self.A).permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)
       
        scale_factorB = self.scaleB(image_embedding)
        shift_factorB = self.shiftB(image_embedding)

        self.B = torch.bmm(self.B1,self.B2)
        
        conditioned_local_feature_matrix = self.B.repeat(batch_size, 1, 1)
        self.eB = conditioned_local_feature_matrix * scale_factorB.view(batch_size, 1, -1) + shift_factorB.view(batch_size, 1, -1)
        self.eB = self.eB.permute(0,2,1)
        
        
        self.load_pad_vectors(self.eA,self.eB,device=x.device)

        base = torch.zeros(x.size(0), 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        
        x_prompted = x + prompt
        return x_prompted
    

    
class PadPrompterBlackBox(nn.Module):
    "Train text prompt initialized matrix and gloabl prompt matrix"
    def __init__(self, args,**kwargs):
        super(PadPrompterBlackBox, self).__init__()
        self.args = args
       
        assert args.TRAINER.NAME in ['LAVIP','BLACKLAVIP'], 'trainer name must be one of LAVIP,BLACKLAVIP'
        trainer_config = {
            'LAVIP': (args.TRAINER.VPWB.PROMPT_SIZE, args.TRAINER.VPWB.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
            'BLACKLAVIP': (args.TRAINER.BLACKLAVIP.PROMPT_SIZE, args.TRAINER.BLACKLAVIP.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
          
        }

        self.pad_size, self.image_size,self.backbone = trainer_config[args.TRAINER.NAME]
        mat_dict = ab_dict[self.image_size]
        vlm_dim= embed_dim_dict[self.backbone]

        self.pad_size, self.image_size,self.backbone = trainer_config[args.TRAINER.NAME]

       
                    

        n_classes = kwargs['n']

        
        m = 8
        r = 32
        
            
        a,b = mat_dict[self.pad_size]
        self.base_size = self.image_size - self.pad_size*2

        self.B1 = nn.Parameter(torch.randn([1,b, r]),requires_grad=True)
        self.B2 = nn.Parameter(torch.randn([1,r, m]),requires_grad=True)
        # self.B = nn.Parameter(torch.randn([1,b, m]),requires_grad=True)
        self.M = nn.Parameter(torch.randn([m, n_classes]),requires_grad=True)
        


        self.fc1 = nn.Linear(vlm_dim, a,bias=False)
    
        self.scaleB = nn.Linear(vlm_dim, m,bias=False)
        self.shiftB = nn.Linear(vlm_dim, m,bias=False)

       
       
       
 

    def load_pad_vectors(self,A, B, device='cpu'):
        
        matrix = torch.bmm(A, B)
        sizes = lambda c, h, w, p: [c * p * h, c * p * h, c * (h - 2 * p) * p, c * (h - 2 * p) * p]
        submatrix_sizes = sizes(3, self.image_size, self.image_size, self.pad_size)
        
        # Unravel the matrix and split it into four parts
        submatrices = torch.split(matrix.view(matrix.size(0),-1), submatrix_sizes, dim=1)

        # Reshape and move the submatrices to the specified device
        self.pad_up = submatrices[0].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_down = submatrices[1].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_left = submatrices[2].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)
        self.pad_right = submatrices[3].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)

        



    def forward(self, x,embedding,image_embedding=None):
        
        embedding = embedding.to(x.device)
        
        image_embedding = image_embedding.to(x.device)
        batch_size = image_embedding.size(0)
        
        
        # self.eA = self.fc1(embedding).permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)
        self.A = self.fc1(embedding)
        self.eA = torch.matmul(self.M,self.A).permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)
       
        scale_factorB = self.scaleB(image_embedding)
        shift_factorB = self.shiftB(image_embedding)

        self.B = torch.bmm(self.B1,self.B2)
        
        conditioned_local_feature_matrix = self.B.repeat(batch_size, 1, 1)
        self.eB = conditioned_local_feature_matrix * scale_factorB.view(batch_size, 1, -1) + shift_factorB.view(batch_size, 1, -1)
        self.eB = self.eB.permute(0,2,1)
        
        
        self.load_pad_vectors(self.eA,self.eB,device=x.device)

        base = torch.zeros(x.size(0), 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        
        x_prompted = x + prompt
        return x_prompted

class PadPrompterB2N(nn.Module):
    "Train text prompt initialized matrix and gloabl prompt matrix"
    def __init__(self, args,**kwargs):
        super(PadPrompterB2N,self).__init__()
        self.args = args
       
        assert args.TRAINER.NAME in ['LAVIP','BLACKLAVIP'], 'trainer name must be one of LAVIP,BLACKLAVIP'
        trainer_config = {
            'LAVIP': (args.TRAINER.LAVIP.PROMPT_SIZE, args.TRAINER.LAVIP.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
            'BLACKLAVIP': (args.TRAINER.BLACKLAVIP.PROMPT_SIZE, args.TRAINER.BLACKLAVIP.IMAGE_SIZE,args.MODEL.BACKBONE.NAME),
          
        }

        self.pad_size, self.image_size,self.backbone = trainer_config[args.TRAINER.NAME]
        mat_dict = ab_dict[self.image_size]
        vlm_dim = embed_dim_dict[self.backbone]
        n_classes = kwargs['n']

        m = 32
        r = 32
        
            
        a,b = mat_dict[self.pad_size]
        self.base_size = self.image_size - self.pad_size*2
        
        


        self.B1 = nn.Parameter(torch.randn([1,b, r]),requires_grad=True)
        self.B2 = nn.Parameter(torch.randn([1,r, m]),requires_grad=True)
        # self.B = nn.Parameter(torch.randn([1,b, m]),requires_grad=True)

        self.scaleB = nn.Linear(vlm_dim, m,bias=False)
        self.shiftB = nn.Linear(vlm_dim, m,bias=False)

        
        print('Subsample Classes: ',self.args.DATASET.SUBSAMPLE_CLASSES)
        if self.args.DATASET.SUBSAMPLE_CLASSES in ['all','base']:
            self.M = nn.Parameter(torch.randn([m, n_classes]),requires_grad=True)
            self.fc1 = nn.Linear(vlm_dim, a,bias=False)
        else:
            self.MA = nn.Parameter(torch.randn([m, a]),requires_grad=True)


    def load_pad_vectors(self,A, B, device='cpu'):
        
        matrix = torch.bmm(A, B)
        sizes = lambda c, h, w, p: [c * p * h, c * p * h, c * (h - 2 * p) * p, c * (h - 2 * p) * p]
        submatrix_sizes = sizes(3, self.image_size, self.image_size, self.pad_size)
        
        # Unravel the matrix and split it into four parts
        submatrices = torch.split(matrix.view(matrix.size(0),-1), submatrix_sizes, dim=1)

        # Reshape and move the submatrices to the specified device
        self.pad_up = submatrices[0].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_down = submatrices[1].reshape(-1, 3, self.pad_size, self.image_size).to(device)
        self.pad_left = submatrices[2].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)
        self.pad_right = submatrices[3].reshape(-1, 3, self.image_size - self.pad_size * 2, self.pad_size).to(device)

        



    def forward(self, x,embedding,image_embedding=None):
        
        embedding = embedding.to(x.device)
        
        image_embedding = image_embedding.to(x.device)
        batch_size = image_embedding.size(0)
        
        
        # self.eA = self.fc1(embedding).permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)
        if self.args.DATASET.SUBSAMPLE_CLASSES in ['all','base']:
            self.A = self.fc1(embedding)
            self.MA = torch.matmul(self.M,self.A)
            self.eA = self.MA.permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)
        
        if self.args.DATASET.SUBSAMPLE_CLASSES=='new':
            p = torch.kron(embedding,self.MA)

            p_reshaped = p.view(self.MA.shape[0], self.MA.shape[1], embedding.shape[0],embedding.shape[1])
            MA_new= p_reshaped.mean(dim=(2,3)).to(x.device)
            self.eA = MA_new.permute(1,0).unsqueeze(0).expand(x.size(0),-1,-1)

        scale_factorB = self.scaleB(image_embedding)
        shift_factorB = self.shiftB(image_embedding)

        self.B = torch.bmm(self.B1,self.B2)
        
        conditioned_local_feature_matrix = self.B.repeat(batch_size, 1, 1)
        self.eB = conditioned_local_feature_matrix * scale_factorB.view(batch_size, 1, -1) + shift_factorB.view(batch_size, 1, -1)
        self.eB = self.eB.permute(0,2,1)
        
        
        self.load_pad_vectors(self.eA,self.eB,device=x.device)

        base = torch.zeros(x.size(0), 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        
        x_prompted = x + prompt
        return x_prompted






    


def padding(args):
    return PadPrompter(args)

def padding_lavip(args,**kwargs):

    return PadPrompterLaViP(args,**kwargs)

def padding_bb(args,**kwargs):
    return PadPrompterBlackBox(args,**kwargs)

def padding_b2n(args,**kwargs):

    return PadPrompterB2N(args,**kwargs)



    

