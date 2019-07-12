# machineSponge
An RNN Implementation designed to generate original SpongeBob episodes from existing transcripts.

(Temp README while project is being finished up)

To use this software, clone or download to a local directory, and run this command in a terminal/CMD Prompt in the downloaded directory (assuming you have Python 3 installed)

Windows: main.py --mode [generate, train]

Before you can generate transcripts, you must TRAIN the model. Therefore, the first step you must take is this command:

Windows: main.py --mode train

Afterwards, you may use the 'generate' argument to create transcripts based off of your newly created model.

NOTE: This current pre-release uses WINDOWS PyTorch for CPUs, and will not make use of CUDA-supported hardware unless modified in the main.py model.
A main.py model that makes use of CUDA will be added soon.

(Idea and base RNN code comes from MortyFire, created by Sarthak Mittal (@naiveHobo)
