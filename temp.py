for i in range(4,31):
	#torch.manual_seed(random.randint(1,10000))
	sample = Variable(torch.randn(64, 20))
	sample = sample.cuda()
	sample = model.decode(sample).cpu()
	sample = sample.data.view(64*28, 28)
	np.save('vae_results/sample_10_trial_' + str(i) + '.npy', sample.numpy())
