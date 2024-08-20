import torch

# from averageMeter import AverageMeter


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CallBacks:
    def __init__(self, best, save_path):
        self.best = best
        self.earlyStop = AverageMeter()
        self.save_path = save_path

    def saveBestModel(self, cur, model):
        if cur > self.best:
            self.best = cur
            torch.save(model.state_dict(), '{}/logs_CRACK500_efficientnet-b3_finnal.pth'.format(self.save_path))
            self.earlyStop.reset()
            print("\n Saving Best Model....\n")
        return

    def earlyStoping(self, cur, maxVal):
        if cur < self.best:
            self.earlyStop.update(1)

        return self.earlyStop.count > maxVal
