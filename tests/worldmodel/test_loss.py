import torch

from OfflineOnline.worldmodel.crossentropy import CrossEntropyLoss

lossFn = CrossEntropyLoss()
latent_size = 5
test_range = 10
true_dist_means = torch.randn(latent_size)*5
true_dist_stds = torch.rand(latent_size)*5
noise_means = torch.randn(latent_size) / test_range

for i in range(test_range):
    meanpd = true_dist_means + noise_means * i
    print(lossFn(true_dist_means, true_dist_stds, meanpd, true_dist_stds))