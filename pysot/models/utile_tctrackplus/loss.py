import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def get_cls_loss(pred, label, select):
    if len(select.shape) == 0 or select.shape == paddle.to_tensor([0]).shape:
        return paddle.to_tensor(0)
    pred = paddle.index_select(pred, 0, select)
    label = paddle.index_select(label, 0, select)
    label = label.astype('int64')
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.reshape([-1, 2])
    label = label.reshape([-1])
    pos = paddle.nonzero(label == 1).squeeze().astype('int64')
    neg = paddle.nonzero(label == 0).squeeze().astype('int64')
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def l1loss(pre,label,weight):
    loss=(paddle.abs((pre-label))*weight).sum()/(weight).sum()
    return loss

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.shape
    pred_loc = pred_loc.reshape([b, 4, -1, sh, sw])
    diff = (pred_loc - label_loc.reshape([b, 4, -1, sh, sw])).abs()
    diff = diff.sum(axis=1).reshape([b, -1, sh, sw])
    loss = diff * loss_weight
    return loss.sum().div(b)

def DISCLE(pred, target, weight):
    pred_x = (pred[:,:, 0]+pred[:,:, 2])/2
    pred_y = (pred[:,:, 1]+pred[:,:, 3])/2
    pred_w = (-pred[:,:, 0]+pred[:,:, 2])
    pred_h = (-pred[:,:, 1]+pred[:,:, 3])

    target_x = (target[:,:, 0]+target[:,:, 2])/2
    target_y = (target[:,:, 1]+target[:,:, 3])/2
    target_w = (-target[:,:, 0]+target[:,:, 2])
    target_h = (-target[:,:, 1]+target[:,:, 3])
    
    loss=paddle.sqrt(paddle.pow((pred_x-target_x),2)/target_w+paddle.pow((pred_y-target_y),2)/target_h)
    weight=weight.reshape(loss.shape)
        
    return  (loss * weight).sum() / (weight.sum()+1e-6)

class IOULoss(nn.Layer):
    def forward(self, pred, target, weight=None):
        
        pred_left = pred[:,:, 0]
        pred_top = pred[:,:, 1]
        pred_right = pred[:,:, 2]
        pred_bottom = pred[:,:, 3]

        target_left = target[:,:, 0]
        target_top = target[:,:, 1]
        target_right = target[:,:, 2]
        target_bottom = target[:,:, 3]

        target_area = (target_right-target_left) * (target_bottom-target_top)
        pred_area = (pred_right-pred_left) * (pred_bottom-pred_top)

        w_intersect = paddle.minimum(pred_right, target_right) - paddle.maximum(pred_left, target_left)
        w_intersect = w_intersect.clip(min=0)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) - paddle.maximum(pred_top, target_top)
        h_intersect = h_intersect.clip(min=0)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = ((area_intersect) / (area_union + 1e-6)).clip(min=0) + 1e-6
        
        losses = -paddle.log(ious)
        weight = weight.reshape(losses.shape)

        return (losses * weight).sum() / (weight.sum() + 1e-6)

class dIOULoss(nn.Layer):
    def forward(self, pred, target, weight=None):
        
        pred_left = pred[:,:, 0]
        pred_top = pred[:,:, 1]
        pred_right = pred[:,:, 2]
        pred_bottom = pred[:,:, 3]

        target_left = target[:,:, 0]
        target_top = target[:,:, 1]
        target_right = target[:,:, 2]
        target_bottom = target[:,:, 3]
        
        prx=((pred_left+pred_right)/2)
        pry=((pred_top+pred_bottom)/2)
        tax=((target_left+target_right)/2)
        tay=((target_top+target_bottom)/2)

        target_area = (target_right-target_left) * (target_bottom-target_top)
        pred_area = (pred_right-pred_left) * (pred_bottom-pred_top)

        w_intersect = paddle.minimum(pred_right, target_right) - paddle.maximum(pred_left, target_left)
        w_intersect = w_intersect.clip(min=0)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) - paddle.maximum(pred_top, target_top)
        h_intersect = h_intersect.clip(min=0)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = ((area_intersect) / (area_union + 1e-6)).clip(min=0) + 1e-6
        
        losses = -paddle.log(ious) + (((prx-tax)**2+(tay-pry)**2)**0.5)*0.2

        weight = weight.reshape(losses.shape)
        if weight.sum() > 0:
            return (losses * weight).sum() / (weight.sum() + 1e-6)
        else:
            return (losses * weight).sum()
  
class gIOULoss(nn.Layer):
    def forward(self, pred, target, weight=None):
        
        pred_left = pred[:,:, 0]
        pred_top = pred[:,:, 1]
        pred_right = pred[:,:, 2]
        pred_bottom = pred[:,:, 3]

        target_left = target[:,:, 0]
        target_top = target[:,:, 1]
        target_right = target[:,:, 2]
        target_bottom = target[:,:, 3]
        
        x1 = paddle.minimum(pred_left, pred_right)
        y1 = paddle.minimum(pred_top, pred_bottom)
        x2 = paddle.maximum(pred_left, pred_right)
        y2 = paddle.maximum(pred_top, pred_bottom)
    
        xkis1 = paddle.maximum(x1, target_left)
        ykis1 = paddle.maximum(y1, target_top)
        xkis2 = paddle.minimum(x2, target_right)
        ykis2 = paddle.minimum(y2, target_bottom)
    
        xc1 = paddle.minimum(x1, target_left)
        yc1 = paddle.minimum(y1, target_top)
        xc2 = paddle.maximum(x2, target_right)
        yc2 = paddle.maximum(y2, target_bottom)
    
        intsctk = paddle.zeros(x1.shape)
        
        mask = (ykis2 > ykis1) * (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (target_right - target_left) * (target_bottom - target_top) - intsctk + 1e-7
        iouk = intsctk / unionk
    
        area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
        miouk = iouk - ((area_c - unionk) / area_c)
        
        losses = 1 - miouk
        weight = weight.reshape(losses.shape)
        if weight.sum() > 0:
            return (losses * weight).sum() / (weight.sum() + 1e-6)
        else:
            return (losses * weight).sum()
