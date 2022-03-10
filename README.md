# Pokedex

Welcome to the Pokédex! Input a .jpg image of any of the first 75 Pokémon registered in the official Pokédex, and our Pokédex application will identify and
display your Pokémon, just like the real thing!

This Pokédex is implemented using transfer learning from a ResNet50 image classifier. We removed the final fully connected layer and replaced it with a sequence of linear and non-linear layers resulting in 75 output channels for the 75 Pokémon our application currently supports. The Pokédex application is launched from Google Colab (using PokédexColab.ipynb) as a website using flask.

Please note only .jpg files are supported at this time. 

This application was created as a project for the University of British Columbia in a team of four including shirleyqi26, Dylan-e, vicky208, and myself.

https://github.com/shirleyqi26

https://github.com/Dylan-e

https://github.com/vicky208

!!Note must have an authenticated ngrok account. 
