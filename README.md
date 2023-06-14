# AIS
基于 DBSAN 的 AIS 航行轨道预测

一：实验流程
	根据给出的船舶航行数据分析船舶航行路线；
  绘制航行路线地图；
  分析船舶运输概况，包括可能的货物、目的地、航行停靠港口；
  利用DBSCAN算法拟合航行路线；
  利用GRU模型进行船舶下一时间点的坐标预测，实现路线预测目的。

二：实验数据
	实验数据是ais_pts.csv，其中每行包含一条AIS信息数据包中的mmsi,basedatetime,lat,lon,
  sog,cog,heading,vesselname,imo,callsign,vesseltype,status,length,width,draft,cargo,transcieverclass属性；
  各属性具体含义如下：
  
mmsi:海上移动业务标识码，是船舶的唯一标识。
basedatetime:AIS数据包的时间戳。
lat:纬度，lon:经度。
sog:船舶的地面速度，单位为节(knots)。
cog:船舶的地面航向即船舶相对地面的运动方向，度数范围为0-359。
heading:船头的朝向。
vesselname:船名。
io:国际海事组织的缩写，指船舶的国际海事组织编码，也是船舶的唯一标识符之一。
callsign:呼号，vesseltype:船舶类型。
status:导航状态：航行、停泊、靠泊、故障无法操纵、因吃水限制受限制、搁浅、捕鱼、保留。
length:船长，width:船宽。
draft:船舶的吃水深度，即船体下沉的深度。
cargo:船体所运输的货物类型。
transciever class:AIS设备的级别或类型，classA与classB相比能传递更多信息。
 

三：实验环境
3.1 实验环境： 
Pandas：2.0.0，用于数据处理；
folium：0.14.0，用于地图绘制；
pyproj：3.5.0，用于地理坐标转换；
datetime：5.1，用于处理日期和时间；
geographiclib：2.0，用于计算地球上大圆线和小圆弧，即进行地理坐标信息的运算；
scikit-learn：1.2.2，用于使用其中的DBSCAN聚类方法将轨迹数据聚类成航线；
numpy：1.24.2，用于处理数值；
seaborn：0.12.2，用于画优雅直观且可显示中文的可视化图像；
matplotlib：3.7.1，用于画图；
torch：2.0.0，用于搭建神经网络模型、对数据进行训练和测试等。

3.2 实验配置：
前期使用华为matebook14 i5（gpu：NVIDIA GeForce，内存2G）；
后期由于数据太大，DBSCAN对轨迹进行聚类时常出现内核停止的现象，故租用了一张RTX 2080 Ti的卡进行实验。





