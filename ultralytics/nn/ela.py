#import torch
#import torch.nn as nn

#class EfficientLocalizationAttention(nn.Module):
  #  def __init__(self,channel,kernel_size=7):
   #     super(EfficientLocalizationAttention,self).__init__()
    #    self.pad = kernel_size // 2
     #   self.conv = nn.Conv1d(channel,channel,kernel_size=kernel_size,padding=self.pad,groups=channel,bias=False)
      #  self.gn = nn.GroupNorm(16,channel)
       # self.sigmoid = nn.Sigmoid()

 #   def forward(self,x):
  #      b, c, h, w = x.size()
#
 #       x_h = torch.mean(x,dim=3,keepdim=True).view(b,c,h)
  #      x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b,c,h,1)
#
 #       x_w = torch.mean(x,dim=2,keepdim=True).view(b,c,w)
  #      x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b,c,w,1)
   #     print('222')
#
 #       return x*x_h*x_w
    
#if __name__ == "__main__":

 #   dummy_input = torch.randn(2,64,32,32)
#
 #   ela = EfficientLocalizationAttention(channel=dummy_input.size(1),kernel_size=7)
##   output = ela(dummy_input)

  #  print(output.shape)
