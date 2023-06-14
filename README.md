# AIS
基于 DBSAN 的 AIS 航行轨道预测

一：实验流程
1. 根据给出的船舶航行数据分析船舶航行路线；
2. 绘制航行路线地图；
3. 分析船舶运输概况，包括可能的货物、目的地、航行停靠港口；
4. 利用DBSCAN算法拟合航行路线；
5. 利用GRU模型进行船舶下一时间点的坐标预测，实现路线预测目的。

二：实验数据
实验数据是ais_pts.csv
>其中每行包含一条AIS信息数据包中的mmsi,basedatetime,lat,lon,sog,cog,heading,vesselname,imo,callsign,vesseltype,status,length,width,draft,cargo,transcieverclass属性

各属性具体含义如下：
- **mmsi**:海上移动业务标识码，是船舶的唯一标识。
- basedatetime:AIS数据包的时间戳。
- **lat**:纬度，**lon**:经度。
- sog:船舶的地面速度，单位为节(knots)。
- cog:船舶的地面航向即船舶相对地面的运动方向，度数范围为0-359。
- **heading**:船头的朝向。
- vesselname:船名。
- io:国际海事组织的缩写，指船舶的国际海事组织编码，也是船舶的唯一标识符之一。
- callsign:呼号。
- vesseltype:船舶类型。
- status:导航状态：航行、停泊、靠泊、故障无法操纵、因吃水限制受限制、搁浅、捕鱼、保留。
- length:船长，width:船宽。
- draft:船舶的吃水深度，即船体下沉的深度。
- cargo:船体所运输的货物类型。
- transciever class:AIS设备的级别或类型，classA与classB相比能传递更多信息。
 

三：实验环境

3.1 实验环境： 
> Pandas：2.0.0，用于数据处理；
> **folium**：0.14.0，用于地图绘制；
> pyproj：3.5.0，用于地理坐标转换；
> datetime：5.1，用于处理日期和时间；
> **geographiclib**：2.0，用于计算地球上大圆线和小圆弧，即进行地理坐标信息的运算；
> scikit-learn：1.2.2，用于使用其中的DBSCAN聚类方法将轨迹数据聚类成航线；
> numpy：1.24.2，用于处理数值；
> seaborn：0.12.2，用于画优雅直观且可显示中文的可视化图像；
> matplotlib：3.7.1，用于画图；
> **torch**：2.0.0，用于搭建神经网络模型、对数据进行训练和测试等。

3.2 实验配置：

前期使用华为matebook14 i5（gpu：NVIDIA GeForce，内存2G）；
后期由于数据太大，DBSCAN对轨迹进行聚类时常出现内核停止的现象，故租用了一张RTX 2080 Ti的卡进行实验。

四:实验过程

4.1 导入数据

1. 将ais_pts.csv放置在代码所在文件夹下，使用”import pandas as pd”将pandas模块命名为pd后，使用pandas库中的read_csv函数将文件导入，命名为df_ais_pts，类型为pandas.core.frame.DataFrame

4.2 数据分析

1. 统计AIS数据包中的空值
2. 统计mmsi种类数（即船只个数）
3. 分别定义如下函数：
   - utcsec2str()：将数据中的utc秒数转换为时间串
   - timestr2sec()：将字符串解析为时间对象,再转换为utc时间戳
   - getTrajByMMSI(mmsi)：用于由mmsi信息得到船只的行迹信息
   - traj_to_folium()：用于将轨迹在地图上可视化
   - distance()：用于计算两个经纬坐标点的地理距离。
   - traj_simplify(traj)：用于去除轨迹数据中时间重复的点，简化轨迹，便于后续计算。
   - traj_stay_merge(traj)：用于计算轨迹的起始时间、终止时间以及该轨迹中所有位置点的平均值，以便后续分析船舶运输概况，包括可能的货物、目的地、航行停靠港口等。
   - traj_stay_point()：用于计算轨迹停留点,返回轨迹停留点列表。
  
4.3 轨迹可视化

1. 船只轨迹可视化
   - 在地图上可视化：结果可见data/output中的plot_traj_mmsi=xx.html，xx对应船只独有的mmsi信息。
   - 在坐标轴上将行迹可视化：结果可见data/output中的AIS轨迹_mmsi.jpg,显示的是船只坐标点的散点图。
2. 船只停留点分析与可视化
   - 计算船只停留点
   - 将停留点可视化，结果可见data/output中的plot_mmsi=xx.html。

4.4 基于DBSCAN聚类得到航行路线

1. 根据航行数据分析船舶航行路线
   将提取出数据中所有的经纬度坐标进行散点图可视化，可看出航行数据中10条船只并不是在一个有关联的相邻区域进行活动的，因此分析船舶航行路线需要每一艘独立进行分析。
2. 对每条船的经纬度信息进行数据预处理，设置不同的eps、min_sample参数进行DBSCAN聚类分析
   - 通过分析聚类后分成的标签种类，标签种类数，数据中无法进行聚类的异常点个数以及每个簇的样本个数，可以评判聚类效果
   - 不断调整DBSAN模型中的参数。
   - 将聚类得到的坐标点的种类标签与坐标点合并成列表，将该列表可视化，标注出每个种类。
  
4.5 搭建GRU模型预测船只路线

1. 整体分析
   
   - 由于船只的轨迹数据中取出了时间戳一样的重复点，因此提取时不能直接调用没有简化过的原始数据，而是将简化过后的经纬度坐标列表position转换为pandas中的DataFrame类型，并将DataFrame的下标设置为简化轨迹中的时间戳。
  
2. 构建数据集
   
   - 将船只坐标点中每60个数据条的前59条作为模型的输入数据，后1条作为输入数据的标签。
   - 将处理好的数据集切分，80%作为训练集，20%作为测试集。
  
3. 搭建GRU模型并进行实例化
4. 训练模型，定义欧几里得算法距离函数算出结果与真实值的相似度，用于自动优化模型。
5. 将训练过程的相似度值和acc值可视化
   
五. 实验结果
   - 对每条船的可视化进行分析
   - 得到可能的状态、可能的货物、目的地、航行停靠港口
1. mmsi=265023000
    - 两个较大的停留点一个在港口，一个在远海某海域，推测应当是一艘远海捕鱼船，进行捕鱼后返回港口。
    - 轨迹图中右侧两小个数据点和左侧坐标点的折线上几乎没有数据点，坐标散点图上也可见
    > 由此推断两侧的数据可能相距时间较远。
- DBSCAN参数：eps=1, min_samples=5
2. mmsi=316001002
    - 由坐标散点图可看出坐标信息点相距较近
    - 由轨迹地图和停留点可看出船只一直在陆地上一个很小的固定区域移动，再由放大后的地图可看出船只在离入海口较近
    > 可推测出是一个货船，且在数据时间范围内一直在装卸货物；或船只正处于维修或休息状态。
- 无需使用DBSCAN聚类出路线和使用GRU模型预测下一时间戳的坐标值。
3. mmsi=338143353
   - 由坐标散点图和轨迹地图可看出船只每隔一定时间就会发送一次信息，一直在匀速运动
   > 推测是一艘货船，停留点显示的是船只在码头的停泊状态，可推测货物是便于保存的、可以接受运输时间较长的电子产品类或其他类别。
- 无需聚类可视化（只有一条路），但可使用GRU预测。
4. mmsi=338999000
   - 由停留点分析图可看出船舶在三个码头来回运输货物，中途有在海洋中央停留
   - 推测是一条近海渔船
5. mmsi= 367348910
    - 由坐标散点图可看出坐标信息点相距较近
    - 由轨迹地图和停留点可看出船只一直停留在码头
    > 推测船只正处于休息状态。
- 无需使用DBSCAN聚类出路线和使用GRU模型预测下一时间戳的坐标值。
6. mmsi=368083770
    - 由坐标散点图可看出坐标信息点分布较均匀，- 由轨迹地图和停留点可看出船只在近海的航道与河道中航行，不断在码头之间来回，不时停留在码头上。
    > 推测这艘船是货船，且运输的物品大概是保质期较短的生鲜类或速冻类，或是家具等需要跨地来回运输的货品。
- eps=0.01, min_samples=20
7. mmsi=368084090
    - 由坐标散点图可看出坐标信息点分布较均匀，- 由轨迹地图和停留点可看出船只在河道中航行，且不断在码头上较短停留。
    > 可推测这艘船是货船，可能是快递公司的货运船，在不同的地方停留卸货和装货。
- 无需聚类可视化（只有一条路），但可使用GRU预测。
8. mmsi=370039000
   - 由坐标散点图可看出坐标信息点分布较均匀但分布较广
   - 由轨迹地图和停留点可看出船只在近海上航行，且在两个码头上往返，在海中央有停留。
    > 可推测这艘船是渔船，在近海捕鱼后送往两地。
- DBSCAN参数：eps=0.1, min_samples=6
9. mmsi= 372628000
    - 由坐标散点图可看出坐标信息点分布较均匀，- 由轨迹地图和停留点可看出船先是在近海上航行，在码头短暂停留一段时间后开向远海，且由信息点越来越稀疏可得知船速在逐渐加快。
    > 可推测这艘船是远海货船，体型较大。
- DBSCAN参数：eps=0.5, min_samples=6
10. mmsi=563045300
    - 由坐标散点图可看出坐标信息点分布较均匀，- 由轨迹地图和停留点可看出船一直在近海上航行，在码头停留了一段时间。
    > 可推测这艘船是近海货船，体型较小。
- 无需聚类可视化（只有一条路），但可使用GRU预测。



