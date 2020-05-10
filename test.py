import matplotlib.pyplot as plt

x = range(1,31)

lstm_val = [0.6772963588775737, 0.6735760059557526, 0.6314522614113737, 0.6259239238342744, 0.6263265988757554, 0.6311971808265034, 0.6260896430539326, 0.6199187544985133, 0.6153807840912989, 0.6228500841172527, 0.6179618752679552, 0.6252290109531231, 0.6123089067376535, 0.6479693235137601, 0.6157554710853791, 0.6095415794766684, 0.6142448874893606, 0.6093692573985664, 0.6025713378763567, 0.6125626354276709, 0.6089162670842576, 0.6020332418333385, 0.6033702093602206, 0.6089071699464634, 0.6003005383511646, 0.6064595044165964, 0.6272971688751646, 0.6323231044640863, 0.6127826533428935, 0.6104148165822949]
conv_lstm_val = [0.6307095209897281, 0.6291452360318621, 0.6306025982019571, 0.6258490005861008, 0.6149355205846044, 0.6363383117780347, 0.6202515336666408, 0.6936999999986393, 0.6637020496915282, 0.6246774536731783, 0.6098678066014521, 0.6095648818522572, 0.6266270696295563, 0.8660069436641612, 0.6127057370523967, 0.7364496916407285, 0.627065505250323, 0.6679554507493111, 0.6376090627558639, 0.6295550191063435, 0.6116031745854075, 0.6237711412261497, 0.6116888088808606, 0.6678266038319635, 0.6091572691021248, 0.6301685086643779, 0.6546476743653014, 0.6860254737716447, 0.6421952350251686, 0.6415785224477555]
nn_val = [0.6483835050766854, 0.6456400445219849, 0.6455647739091483, 0.6462681576745037, 0.6521665359083932, 0.6454359166819971, 0.6498182366917842, 0.6541478512376375, 0.6764973158956588, 0.6810776827203252, 0.6965897701346465, 0.7129060809684268, 0.7193078662915091, 0.7385746658366761, 0.782564960313102, 0.7833567976800155, 0.8719486929181384, 0.8718740411090394, 0.8918472008074668, 0.951294658517926, 1.0115717665652637, 1.0026999881374024, 1.0981582021778493, 1.128714078299568, 1.1601806872403366, 1.1964986197680285, 1.222597413138538, 1.2962301928871458, 1.3447693959715825, 1.3608609573892119]

stack_lstm_val = [0.6337705712554672, 0.6199075922196671, 0.6123021990337482, 0.6087161006727585, 0.6080397282028366, 0.6035997574657659, 0.6025327705316479, 0.5987044275360327, 0.6043692150156041, 0.5983433582316067, 0.594979679474998, 0.6011448203498033, 0.6003625219569322, 0.5953804927115495, 0.5969274012350237, 0.606779693124302, 0.6072332466626044, 0.6019682703940362, 0.6187470576216326, 0.5991785443797205, 0.6003593089852435, 0.598079799056391, 0.6156516226461226, 0.6161193153754039, 0.6148557822503711, 0.6146850561997786, 0.6225489697404636, 0.6519678868029365, 0.6261740075194054, 0.618489449358308]
'''
plt.plot(x[:20],lstm_val[:20],color='red',label='LSTM loss')
plt.plot(x[:20],conv_lstm_val[:20],color='green',label='RCNN loss')

plt.plot(x[:20],nn_val[:20],color='blue',label='ANN loss')
plt.plot(x[:20],stack_lstm_val[:20],color='purple',label='Stack LSTM loss')

'''

lstm_acc = [0.5044463872909546, 0.5410925149917603, 0.6477084159851074, 0.651421844959259, 0.6479038596153259, 0.6534740328788757, 0.6522036790847778, 0.6538649201393127, 0.6585556268692017, 0.6503469347953796, 0.661096453666687, 0.6491742134094238, 0.6637349724769592, 0.651421844959259, 0.6706733107566833, 0.6736050248146057, 0.671552836894989, 0.6647121906280518, 0.6722368597984314, 0.6742890477180481, 0.6768298745155334, 0.6767321228981018, 0.6718459725379944, 0.6717482805252075, 0.6790775060653687, 0.6741913557052612, 0.6422358751296997, 0.671943724155426, 0.6804456114768982, 0.6823023557662964]
rcnn_acc = [0.6435062885284424, 0.6459493637084961, 0.6266002058982849, 0.6456562280654907, 0.6560148596763611, 0.6239616870880127, 0.6579692959785461, 0.5513534545898438, 0.594742476940155, 0.6395973563194275, 0.6631486415863037, 0.660314679145813, 0.6672530174255371, 0.6407700777053833, 0.6628554463386536, 0.5524284243583679, 0.6461448073387146, 0.6596305966377258, 0.6374474763870239, 0.6463402509689331, 0.6628554463386536, 0.6598260402679443, 0.6649076342582703, 0.622300386428833, 0.6638327240943909, 0.6648099422454834, 0.6149711608886719, 0.6680347919464111, 0.6640281677246094, 0.6633440852165222]
stack_lstm_acc = [0.6448744535446167, 0.6584579348564148, 0.6693052053451538, 0.6693052053451538, 0.6682302355766296, 0.6726277470588684, 0.6690120100975037, 0.6809342503547668, 0.671161949634552, 0.6760480999946594, 0.6798592805862427, 0.6752662658691406, 0.6742890477180481, 0.6822046041488647, 0.6855272054672241, 0.6781002879142761, 0.6745822429656982, 0.682400107383728, 0.6779048442840576, 0.6848431825637817, 0.6875793933868408, 0.6859180927276611, 0.6791751980781555, 0.6908042430877686, 0.6841591000556946, 0.6855272054672241, 0.6828886866569519, 0.6716505289077759, 0.6877748370170593, 0.6887520551681519]



plt.plot(x[:20],lstm_acc[:20],color='red',label='LSTM accuracy')
plt.plot(x[:20],rcnn_acc[:20],color='green',label='RCNN accuracy')

plt.plot(x[:20],stack_lstm_acc[:20],color='purple',label='Stack LSTM accuracy')

plt.title('')
plt.xlabel('epoch')
plt.ylabel('')

plt.legend()

plt.show()