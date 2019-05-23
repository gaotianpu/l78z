import mcpi.minecraft as minecraft
import mcpi.block as block 

#Connect to minecraft by creating the minecraft object
# - minecraft needs to be running and in a game
mc = minecraft.Minecraft.create()

#Post a message to the minecraft chat window 
mc.postToChat("Hello,Minecraft")

