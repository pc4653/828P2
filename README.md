# 828P2
### Looking at generative model as a replacement for bootstrap in Alexandrov's mutation signature pipeline

TO DO:

~~Change the set up from wide one FC layer to deep layers;~~ **done**

~~Change the set up from having the input to be sequence of data (2 dimension) to just data (1 dimension)~~ **done**

Adding dropout to the network; necessary?

Generative different sets of signatures to see if similairty between signatures give problem for the network and that if bootstrap actually helps;

Investigate the discovered feature/latent space represented by the decoder - does it have the linearity that other space have? if so we can try playing with arthrimatics in feature space - if feature vector of 5 mutations sigs - feature vector of 2 mutation sigs from the 5 = feature vector of 3 
 
denosing effect

Investigate Poisson noise resistance of the input data


### Variantional Autoencoder

this picture explains variantional autoencoder pretty well: 

![Alt text](vae.4.png?raw=true "Optional Title")

The idea is that, to make an autoencoder a generative process - and not just a decoder recovering from a encoded, existing data, which is what autoencoder does - we try to force the encoded message z to have a certain distribution - gaussian in this implementation - and instead of giving the decoder the encoded message, we can sample from the distribution and feed the generated result to the decoder. After training the decoder should be able to capture the hidden processes that give rise to true data and generate data that's realistic - in the sense that it follows the hidden processes, but also different from input data/not pure memorization but capture of semantic properties. 

11/2/17

trying on biological dataset:

**existing program running on CLL dataset, genome; comparing against 30 COSMIC mutational signatures, K = 5, list of the highest cos_sim matching for each extracted signature and their corresponding COSMIC signature:**

[[0.94206813080597818, 4], [0.87352351453272581, 7], [0.85608103936427404, 16], [0.83670793486890338, 0], [0.92520002216815989, 8]]

eliminating categories: 
[2, 6, 26, 30, 50, 54, 74, 78]
completed in 54.9711399078seconds.
the average Frobenius reconstruction error is: 
285.602870906
the forbenius reconstruction error for the set of estimated P is: 
216.869202261
the average silhoutte width for each cluster is:
[0.92528858222351629, 0.98029434757548839, 0.93919698879810853, 0.91311794141577907, 0.9559143614186858]
average silhoutte width for all is: 
0.942762444286

verified by running Alexandrov's verision
[[0.82643526640751763, 0], [0.85699061686149625, 16], [0.95533379746958702, 8], [0.92132072845980695, 4], [0.88829475099483424, 7]]

**running on K = 3**
my program:
[[0.9221663904759364, 8], [0.86642080588138959, 4], [0.78141173116810625, 15]]
Alex's program:
[[0.88301672842474621, 4], [0.9308140924199525, 8], [0.81143735789793014, 7]]


**K = 3, only exomes:**
[[0.86615684681403948, 4], [0.92160093360522988, 8], [0.78209752802091781, 15]]
eliminating categories: 
[6, 26, 30, 50, 54, 69, 74, 78]
completed in 50.3525850773seconds.
the average Frobenius reconstruction error is: 
317.900581079
the forbenius reconstruction error for the set of estimated P is: 
247.125600979
the average silhoutte width for each cluster is:
[0.99619590010774572, 0.98876095452610013, 0.98765062028567407]
average silhoutte width for all is: 
0.990869158307

**K = 3, using both exomes and genomes**
[[0.86615684681403948, 4], [0.92160093360522988, 8], [0.78209752802091781, 15]]

on Alex's program:
[[0.98446355941428831, 8], [0.85230442708625886, 0], [0.90165568102368965, 4]]

10/27/17

Thoughts on results - training on more and more epochs seem to make the signatures converge better and better - what happens when we try to run on higher/lower order of signatures (presumed K), in Alexandrov's pipeline?


changed the network to deeper layers - 

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 200)		
        self.fc3 = nn.Linear(200, 100)		
        self.fc4 = nn.Linear(100, 50)	
        self.fc51 = nn.Linear(50, 20)
        self.fc52 = nn.Linear(50, 20)
        self.fc6 = nn.Linear(20, 50)		
        self.fc7 = nn.Linear(50, 100)
        self.fc8 = nn.Linear(100, 200)
        self.fc9 = nn.Linear(200, 300)
        self.fc10 = nn.Linear(300, 784)

**results after 400 epochs, data size 10000 **

generated data

![Alt text](10000_gen_data_400.jpg?raw=true "Optional Title")

original data

![Alt text](10000_orig_data.jpg?raw=true "Optional Title")

signatures extracted

![Alt text](10000_sig_extract_comp_400.jpg?raw=true "Optional Title")


**ran 10 epochs**

eliminating categories: 

[2, 4, 8, 13, 17, 23]

completed in 91.2932219505seconds.

the average Frobenius reconstruction error is: 

250.123151126

the forbenius reconstruction error for the set of estimated P is: 

113.291217854

the average silhoutte width for each cluster is:

[0.97317942519728728, 0.95888833995949341, 0.99324184588029374, 0.96146391912447982, 0.90907985501430033]

average silhoutte width for all is: 

0.959170677035

0 signature has the highest similarity with 2 signatures with 0.914057500834

1 signature has the highest similarity with 0 signatures with 0.400483218069

2 signature has the highest similarity with 0 signatures with 0.648148315138

3 signature has the highest similarity with 1 signatures with 0.745043625485

4 signature has the highest similarity with 4 signatures with 0.75558078918

**epoch of 20**

eliminating categories: 

[2, 8, 13, 17, 23]

completed in 84.2069511414seconds.

the average Frobenius reconstruction error is: 

217.554456989

the forbenius reconstruction error for the set of estimated P is: 

51.9703338518

the average silhoutte width for each cluster is:

[0.9681127423503757, 0.83957639540233464, 0.9926201945643951, 0.96932271398756542, 0.95600790510493383]

average silhoutte width for all is: 

0.945127990282

0 signature has the highest similarity with 2 signatures with 0.966877865327

1 signature has the highest similarity with 1 signatures with 0.573454489115

2 signature has the highest similarity with 1 signatures with 0.675458856412

3 signature has the highest similarity with 4 signatures with 0.954165892304

4 signature has the highest similarity with 1 signatures with 0.620786225848

**epoch of 100**


eliminating categories: 

[2, 8, 13, 17, 23]

completed in 84.2080330849seconds.

the average Frobenius reconstruction error is: 

240.570621828

the forbenius reconstruction error for the set of estimated P is: 

62.2400998246

the average silhoutte width for each cluster is:

[0.96033114853392998, 0.99157391376586734, 0.94157361550155116, 0.95633805326259891, 0.96884173678881913]

average silhoutte width for all is: 

0.963731693571

0 signature has the highest similarity with 1 signatures with 0.971470672793

1 signature has the highest similarity with 0 signatures with 0.706407621673

2 signature has the highest similarity with 0 signatures with 0.763335190934

3 signature has the highest similarity with 2 signatures with 0.979331220478

4 signature has the highest similarity with 3 signatures with 0.884240536684

**epoch 200**


eliminating categories: 

[2, 8, 13, 17, 23]

completed in 86.8884019852seconds.

the average Frobenius reconstruction error is: 

291.662053621

the forbenius reconstruction error for the set of estimated P is: 

73.4872082156

the average silhoutte width for each cluster is:

[0.95574083607183036, 0.94643915971959958, 0.93522009006001761, 0.98790781718393805, 0.99322479400027242]

average silhoutte width for all is: 

0.963706539407

0 signature has the highest similarity with 3 signatures with 0.977145904952

1 signature has the highest similarity with 2 signatures with 0.816591532644

2 signature has the highest similarity with 2 signatures with 0.717726167306

3 signature has the highest similarity with 4 signatures with 0.984693958855

4 signature has the highest similarity with 0 signatures with 0.946413400914

**epoch 400**


eliminating categories: 

[2, 8, 13, 17, 23]

completed in 91.1732280254seconds.

the average Frobenius reconstruction error is: 

351.193654548

the forbenius reconstruction error for the set of estimated P is: 

88.2993497315

the average silhoutte width for each cluster is:

[0.95983745591663772, 0.95311246726479981, 0.9952849974127802, 0.98408598591955587, 0.95501282893050576]

average silhoutte width for all is: 

0.969466747089

0 signature has the highest similarity with 3 signatures with 0.988627238989

1 signature has the highest similarity with 1 signatures with 0.951031220307

2 signature has the highest similarity with 4 signatures with 0.970012829318

3 signature has the highest similarity with 2 signatures with 0.986920186276

4 signature has the highest similarity with 0 signatures with 0.966446345334




10/26/17

Progress:

Adjusted the VAE structure to use ReLu as the output activation, and discarded normalizing data to (0,1); use MSE instead of BCE as loss function. Results significantly better: 



loading trial5

eliminating categories: 

**[0, 1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 17, 18, 19, 22, 25]**

completed in 76.3571209908seconds.

the average Frobenius reconstruction error is: 

93.1936229075

the forbenius reconstruction error for the set of estimated P is: 

28.8218693684

the average silhoutte width for each cluster is:

[0.99358637123586324, 0.99986549987679763, 0.97224522168892225, 0.99906020406738838, 0.99177668641782157]

average silhoutte width for all is: 

0.991306796657

0 signature has the highest similarity with 1 signatures with 0.996495409317

1 signature has the highest similarity with 3 signatures with 0.982332194155

2 signature has the highest similarity with 4 signatures with 0.938770560997

3 signature has the highest similarity with 2 signatures with 0.889885039796

4 signature has the highest similarity with 0 signatures with 0.970037641829

loading trial6

eliminating categories: 

**[0, 1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 17, 18, 19, 22, 25]**

completed in 77.1132860184seconds.

the average Frobenius reconstruction error is: 

93.1494199704

the forbenius reconstruction error for the set of estimated P is: 

27.7542537708

the average silhoutte width for each cluster is:

[0.99088208973591541, 0.99986423269803337, 0.99891561071207435, 0.99345008606344309, 0.97417776494092223]

average silhoutte width for all is: 

0.99145795683

0 signature has the highest similarity with 1 signatures with 0.996563357995

1 signature has the highest similarity with 2 signatures with 0.983287405655

2 signature has the highest similarity with 0 signatures with 0.934483656216

3 signature has the highest similarity with 4 signatures with 0.885670381439

4 signature has the highest similarity with 3 signatures with 0.967717413551

loading trial7

eliminating categories: 

**[0, 1, 2, 3, 5, 7, 8, 10, 12, 15, 16, 17, 18, 19, 22, 25]**

completed in 77.3325209618seconds.

the average Frobenius reconstruction error is: 

95.6895897243

the forbenius reconstruction error for the set of estimated P is: 

29.0813072572

the average silhoutte width for each cluster is:

[0.99876252231285578, 0.98912497063350469, 0.9998496604514967, 0.97216836523485928, 0.99281361284712266]

average silhoutte width for all is: 

0.990543826296

0 signature has the highest similarity with 2 signatures with 0.996403704778

1 signature has the highest similarity with 0 signatures with 0.982918340407

2 signature has the highest similarity with 1 signatures with 0.93260345711

3 signature has the highest similarity with 3 signatures with 0.889215443288

4 signature has the highest similarity with 4 signatures with 0.967950686156

### comparing the signautres extracted from generated data to the simulated data's true signatures:

![Alt text](1960000_sig_extract_com.jpg?raw=true "Optional Title")


### Observations:

1. The extracted signatures on generated data are close to the true signatures; however, not absurdly close compare to NMF on bootstrapped original data.

2. The neural network/decoder-encoder is able to learn about the latent structure of the orginal data, to further validate this, let's look into the original data and generated data

Here is a picture of the input data from subplot 2 - 9, the signatures are plotted on 1:
![Alt text](1960000_orig_com.jpg?raw=true "Optional Title")

and here is the picture of the generated data:
![Alt text](1960000_gen_data.jpg?raw=true "Optional Title")


We can see that the genereated data is very similar to unit addition amongst the true signatures - they are also obviously not simple resampling/bootstrapping results. Clearly this decoder network 'recognizes' that certain mutation categories do not play a strong role in the mutation signature's shape, and thus didn't generate a lot of those mutations. This is why in the dimension reduction stage of Alexandrov's pipeline, we are seeing 16 of the 28 mutation categories dropped. 



**3. interesting observation, bootstrapping actually seems to be making a difference - not sure why, maybe the generated signatures are too close together? - further investigation needed **

**here is results without bootstrap used, can see that some of the signatures are not seperated, although the overall cos similairty is still relatively high**

loading trial4

eliminating categories: 

[ 0 12  7 25  8  2 18 17  3 22 19 15  5 16 10  1]

completed in 94.978374958seconds.

the average Frobenius reconstruction error is: 

24.9665067229

the forbenius reconstruction error for the set of estimated P is: 

24.5598110763

the average silhoutte width for each cluster is:

[0.99820912368538284, 0.99589283583106181, 0.9666613359499524, 0.93905834339004868, 0.91086467659416137]

average silhoutte width for all is: 

0.96213726309

0 signature has the highest similarity with 0 signatures with 0.997199036847

**1 signature has the highest similarity with 1 signatures with 0.929217521427**

2 signature has the highest similarity with 3 signatures with 0.913586148252

**3 signature has the highest similarity with 1 signatures with 0.93963581449**

4 signature has the highest similarity with 2 signatures with 0.85521383294


loading trial5

eliminating categories: 

[ 0 18 22  7 16  2 25  3 12 15 19  8 17  5 10  1]

completed in 94.5966670513seconds.

the average Frobenius reconstruction error is: 

25.9204855938

the forbenius reconstruction error for the set of estimated P is: 

25.6733155694

the average silhoutte width for each cluster is:

[0.99575377552995503, 0.93759133125873895, 0.99820779076242694, 0.96841685335593208, 0.9163495487733827]

average silhoutte width for all is: 

0.963263859936

0 signature has the highest similarity with 2 signatures with 0.997020846538

**1 signature has the highest similarity with 0 signatures with 0.928453280473**

2 signature has the highest similarity with 1 signatures with 0.908710360713

**3 signature has the highest similarity with 0 signatures with 0.939580598993**

4 signature has the highest similarity with 3 signatures with 0.863728100932


**results from running on original data, with the same data size, for reference**

eliminating categories: 

[0, 2, 3, 8, 12, 18]

completed in 150.119699955seconds.

the average Frobenius reconstruction error is: 

389.595884234

the forbenius reconstruction error for the set of estimated P is: 

277.120902096

the average silhoutte width for each cluster is:

[0.99928324887360109, 0.98220648040693792, 0.99970417960330649, 0.99962217607270742, 0.99942960264402647]

average silhoutte width for all is: 

0.99604913752

0 signature has the highest similarity with 2 signatures with 0.999366231423

1 signature has the highest similarity with 4 signatures with 0.99867606782

2 signature has the highest similarity with 0 signatures with 0.998940692117

3 signature has the highest similarity with 1 signatures with 0.967777658719

4 signature has the highest similarity with 3 signatures with 0.999391536424


### method implemented on less simulated data - 10000 observations, still 5 signatures and 28 categories

some data on training the network: 500 epochs, batch size 64, input size 28 by 28

====> Epoch: 500 Average train set loss: 0.3672

====> Test set loss: 0.5556

**didn't implement dropout, may be overfitted - futher investigation needed**

loading trial0

eliminating categories: 

[2, 4, 8, 9, 13, 15, 17, 23]

completed in 67.1249649525seconds.

the average Frobenius reconstruction error is: 

78.0202843722

the forbenius reconstruction error for the set of estimated P is: 

55.7013836045

the average silhoutte width for each cluster is:

[0.90775136297211934, 0.9994543002705446, 0.98414278073086292, 0.9970044912216558, 0.98195369249509368]

average silhoutte width for all is: 

0.974061325538

0 signature has the highest similarity with 1 signatures with 0.91110217395

**1 signature has the highest similarity with 4 signatures with 0.276743053483**

2 signature has the highest similarity with 4 signatures with 0.882629638038

3 signature has the highest similarity with 0 signatures with 0.948479500411

4 signature has the highest similarity with 3 signatures with 0.845860815882


signatures comparison, per usual 

![Alt text](10000_sig_extract_comp.jpg?raw=true "Optional Title")

observation: 

Mostly still have good result, consider the sample size is a lot smaller, specifically one of the signature was not found. Can look into the possibility of why.

Even when a signature is not matched, the overall cluster compactness is still not changed - probably means that the decoder uses some other process to generate data, which is found in NMF


10/25/17

Looking into Variantional Autoencoder: 
following pytorch's example on VAE, replacing the input data from MNIST to simulated data, generated by the metrics of 28 categories and N samples, 5 signatures. 

input was set to be 28 by 28, assembling 28 sequences of 28 length into the same size as MNIST 

**Note 1. MNIST was normalized to 0 and 1 (sigmoid was used as the output activation) - whether normalizing simulated data to 0 and 1 is questionable - in general most values are closer to 0, probably better to use another scheme**

Note 2. Whether treating simulated data as sequences is useful is questionable? At the least didn't expect worse output than before but may be wrong

Note 3. Current encoder and decoder uses one wide fc layer before activation, prob better performance on deeper layer

Note 4. Variantional Autoencoder gives a generative model, probably more sane to simply use autoencoder to make sure the general concept works first.

Note 5. try WGAN later

Results so far:

Interestingly engouh, there is some mix result with VAE - here is the log for pipeline running over genereated data:

1st run:

the average Frobenius reconstruction error is: 

65.1315292319

the forbenius reconstruction error for the set of estimated P is: 

16.4943449169

the average silhoutte width for each cluster is:

[0.75799929550017031, 0.99719044244052579, 0.74098777126736559, 0.72619768940514817, 0.71008715858294302]

average silhoutte width for all is: 

0.786492471439

0 signature has the highest similarity with 1 signatures with 0.993470673477

1 signature has the highest similarity with 4 signatures with 0.315196501982

2 signature has the highest similarity with 3 signatures with 0.414266812421

3 signature has the highest similarity with 4 signatures with 0.313893276248

4 signature has the highest similarity with 2 signatures with 0.300356473917



the average Frobenius reconstruction error is: 

14.8659022096

the forbenius reconstruction error for the set of estimated P is: 

1.73926966441

the average silhoutte width for each cluster is:

[0.53235868874473036, 0.97550448331200457, 0.56557660716190294, 0.62100938952537899, 0.74031928449558515]

average silhoutte width for all is: 

0.686953690648

0 signature has the highest similarity with 1 signatures with 0.991926408006

1 signature has the highest similarity with 0 signatures with 0.345187328837

2 signature has the highest similarity with 3 signatures with 0.50602063159

3 signature has the highest similarity with 2 signatures with 0.308905319654

4 signature has the highest similarity with 2 signatures with 0.331819927354


The first signature gets recovered exactly, but not the rest, why? 





