import matplotlib.pyplot as plt


x1 = range(1,9)
x_df = range(1,8)
baseline = [0]*7

# Losses=[0.2050402482351, 0.09714470793803533, 0.0840319378177325, 0.07818545053402583, 0.07341804474592209, 0.06999305908878645, 0.06765230029821395, 0.06498128190636634]
# plt.plot(x1,Losses, label="Exp Losses", linewidth=2,color= 'r', marker='o',markersize=5)
# plt.legend()
# plt.xlabel('n epochs')
# plt.ylabel('Training Loss of Each Epoch')
# plt.show()
#
# Errors=[0.2660150666666667, 0.16559453333333332, 0.14702293333333333, 0.13640639999999998, 0.1308724, 0.12600866666666669, 0.12278320000000001, 0.11836426666666668]
# plt.plot(x1,Errors, label="Errors", linewidth=2,color= 'g', marker='o',markersize=5)
# plt.legend()
# plt.xlabel('n epochs')
# plt.ylabel('Training Error of Each Epoch')
# plt.show()
#
N_classes_Accuracy=[[0.2344706168530898, 0.5615318247995533, 0.7917841837374522], [0.6173679454219823, 0.7336090460599876, 0.8751242113941271], [0.6550243975033054, 0.7736239785059887, 0.8863838851975072], [0.6603198658963594, 0.7969970485973614, 0.8940999681451542], [0.668063905341013, 0.807526565132287, 0.8999345532761123], [0.6660109909416154, 0.8196604846160053, 0.9043560903712754], [0.6740693807370072, 0.8264963736973324, 0.9071106812518042], [0.6823376966488877, 0.8376204427757399, 0.9107441074009207]]
red_hair_acc = [0]*8
blue_back_acc=[0]*8
green_skin_acc = [0]*8

red_hair_acc_df = [0]*7
blue_back_acc_df=[0]*7
green_skin_acc_df = [0]*7

for i in range(8):
    red_hair_acc[i]=N_classes_Accuracy[i][0]
    blue_back_acc[i] = N_classes_Accuracy[i][1]
    green_skin_acc[i] = N_classes_Accuracy[i][2]

    if(i<1):
        continue
    red_hair_acc_df[i-1]=red_hair_acc[i]-red_hair_acc[i-1]
    blue_back_acc_df[i-1]=blue_back_acc[i]-blue_back_acc[i-1]
    green_skin_acc_df[i-1]=green_skin_acc[i]-green_skin_acc[i-1]

plt.title("Accuracy of Three Classes: Hair, Skin and Backgroud")
plt.plot(x1,red_hair_acc, label="Hair", linewidth=1,color= 'r', marker='o',markersize=6)
plt.plot(x1,blue_back_acc, label="Background", linewidth=1,color= 'b', marker='o',markersize=6)
plt.plot(x1,green_skin_acc, label="Skin", linewidth=1,color= 'g', marker='o',markersize=6)
plt.legend()
plt.xlabel('n epochs')
plt.ylabel('Training Accuracy of Each Class')
plt.show()

plt.title("Accuracy Change of Three Classes: Hair, Skin and Backgroud")
plt.plot(x_df,red_hair_acc_df, label="Hair", linewidth=1,color= 'r', marker='o',markersize=6)
plt.plot(x_df,blue_back_acc_df, label="Background", linewidth=1,color= 'b', marker='o',markersize=6)
plt.plot(x_df,green_skin_acc_df, label="Skin", linewidth=1,color= 'g', marker='o',markersize=6)
plt.plot(x_df,baseline, label="Baseline", linewidth=0.5,color= "black", linestyle='--')
plt.legend()
plt.xlabel('n epochs')
plt.ylabel('Training Accuracy Difference')
plt.show()

# -------------Test results
# Test_loss = [0.09519248872995377, 0.1064225235581398, 0.0738830417394638, 0.06792500242590904, 0.06382502138614654, 0.0624775967001915, 0.05825162813067436, 0.056851691007614134]
# plt.plot(x1, Test_loss, label="Test Losses", linewidth=2, color='r', marker='o', markersize=5)
# plt.legend()
# plt.xlabel('n epochs')
# plt.ylabel('Testing Loss of Each Epoch')
# plt.show()
#
# Test_error = [0.17477000000000004, 0.17235119999999995, 0.138626, 0.1324672, 0.1250144, 0.12245800000000001, 0.1135412, 0.11539599999999998]
# plt.plot(x1,Test_error, label="Errors", linewidth=2,color= 'g', marker='o',markersize=5)
# plt.legend()
# plt.xlabel('n epochs')
# plt.ylabel('Testing Error of Each Epoch')
# plt.show()

N_classes_Accuracy_test = [[0.02328125, 0.7882391213799306, 0.8341236643766438], [0.6183117709093273, 0.6204973007643368, 0.93235058315491], [0.6692297962687705, 0.7373128427967686, 0.915270353245206], [0.6803756354990168, 0.7513197098918589, 0.9183220832022243], [0.7171701391531577, 0.840544717269669, 0.8921888324031579], [0.7462400831054657, 0.8244240291497796, 0.8988241470578532], [0.6783489839551567, 0.8443997557512342, 0.9154232841578134], [0.7763689027245797, 0.8519526262515843, 0.899177028451774]]

test_red_hair_acc = [0]*8
test_blue_back_acc=[0]*8
test_green_skin_acc = [0]*8

test_red_hair_acc_df = [0]*7
test_blue_back_acc_df=[0]*7
test_green_skin_acc_df = [0]*7
for i in range(8):
    test_red_hair_acc[i]=N_classes_Accuracy_test[i][0]
    test_blue_back_acc[i] = N_classes_Accuracy_test[i][1]
    test_green_skin_acc[i] = N_classes_Accuracy_test[i][2]

    if(i<1):
        continue
    test_red_hair_acc_df[i-1]=test_red_hair_acc[i]-test_red_hair_acc[i-1]
    test_blue_back_acc_df[i-1]=test_blue_back_acc[i]-test_blue_back_acc[i-1]
    test_green_skin_acc_df[i-1]=test_green_skin_acc[i]-test_green_skin_acc[i-1]

plt.title("Accuracy of Three Classes: Hair, Skin and Backgroud")
plt.plot(x1,test_red_hair_acc, label="Hair", linewidth=1,color= 'r', marker='o',markersize=6)
plt.plot(x1,test_blue_back_acc, label="Background", linewidth=1,color= 'b', marker='o',markersize=6)
plt.plot(x1,test_green_skin_acc, label="Skin", linewidth=1,color= 'g', marker='o',markersize=6)
plt.legend()
plt.xlabel('n epochs')
plt.ylabel('Testing Accuracy of Each Class')
plt.show()

plt.title("Accuracy Change of Three Classes: Hair, Skin and Backgroud")
plt.plot(x_df,test_red_hair_acc_df, label="Hair", linewidth=1,color= 'r', marker='o',markersize=6)
plt.plot(x_df,test_blue_back_acc_df, label="Background", linewidth=1,color= 'b', marker='o',markersize=6)
plt.plot(x_df,test_green_skin_acc_df, label="Skin", linewidth=1,color= 'g', marker='o',markersize=6)
plt.plot(x_df,baseline, label="Baseline", linewidth=0.5,color= "black", linestyle='--')
plt.legend()
plt.xlabel('n epochs')
plt.ylabel('Testing Accuracy Difference')
plt.show()