# machineSponge
An RNN Implementation designed to generate original SpongeBob episodes from existing transcripts.

(Temp README while project is being finished up)

To use this software, clone or download to a local directory, and run this command in a terminal/CMD Prompt in the downloaded directory (assuming you have Python 3 installed)

Windows: main.py --mode [generate, train]

Before you can generate transcripts, you must TRAIN the model. Therefore, the first step you must take is this command:

Windows: main.py --mode train

Afterwards, you may use the 'generate' argument to create transcripts based off of your newly created model.

NOTE: This current pre-release uses WINDOWS PyTorch, choose the proper MAIN file depending whether you wish to use CUDA-supported hardware to expedite the run or not.

(Idea and base RNN code comes from MortyFire, created by Sarthak Mittal (@naiveHobo)
