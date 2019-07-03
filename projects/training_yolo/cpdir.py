import os 
import sys 

# vid = sys.argv[1] # '11267580740956354952'
# os.system("mkdir " + vid)
# os.system("cp output/Annotations/%s* ./%s/" % (vid,vid) )
# os.system("cp output/JPEGImages/%s* ./%s/" % (vid,vid)  )

# path="output/"
# for i, fname in enumerate(os.listdir(path)):


# 1648335475137519817 2850
# find test/ -name "*.jpg" -exec cp {} train \;

os.system("rm -fR output/tmp_img  && mkdir output/tmp_img")
vid_list = '7391956694714582492,9743959100359599877,1998344078699277008,17691859873086356873,11706064642013423526,11348307408514767321,5304831991366008655,17913679268743009861,2450651752444776067,14658145229493843398,5569561346257896546,11414799309983321633,3136212439743070648,11440182015728612525,3428328515596714740,5961197454369096300,6056867447569275258,6419429902125639112,11550075360272050946,8634526054062594525,9610566743018626523,1480081488797101134,5300469961682276990,10111518654269549303,4407481455277765842,642013767377497558,9984875366477005430,15358590448193952574,15818795287108391133,8348653294780138623,8299295122846805979,7220219570634995062,3082489580432728482,6763488444784005156,3474952445196391367,15450878111534878710,17999411403859236964,16003208693148860700,7190721056500514205,11267580740956354952'.split(',')
for vid in vid_list:
    # os.system("find output/Annotations/ -name '%s*.xml' -exec cp {} output/tmp_img/ \;"% (vid))
    os.system("find output/JPEGImages/ -name '%s*.png' -exec cp {} output/tmp_img_1/ \;"% (vid))
    os.system("find output/JPEGImages/ -name '%s*.xml' -exec cp {} output/tmp_img_1/ \;"% (vid))
    # os.system("cp output/Annotations/%s* ./tmp_img/" % (vid) )
    # os.system("cp output/JPEGImages/%s* ./tmp_img/" % (vid)  )



# find output/tmp_img -name '*.xml' -exec cp {}  output/Annotations \;
# find tmp_img -name '5597743532691308942*.xml' -exec cp {} x \; 

# find output/Annotations -name '5597743532691308942*.xml' -exec cp {} x \;
# 
# find output/JPEGImages -name '*.xml' -exec  {} 

# find images -name '*.xml' -exec cp {} tmp_xml/ \;