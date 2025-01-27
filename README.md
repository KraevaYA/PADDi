# PADDi: PALMAD-based Anomaly Discovery on Distributed GPUs
This repository is related to the PADDi (PALMAD-based Anomaly Discovery on Distributed GPUs) algorithm that discovers arbitrary-length discords in a very long time series on a high-performance cluster with nodes, each of which is equipped with multiple GPUs. PADDi is authored by Yana Kraeva (kraevaya@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PADDi's source code (in C, CUDA), accompanying datasets, and experimental results. Please cite an article that describes PADDi as shown below.

Our study is based on the discord concept by Keogh et al. [https://doi.org/10.1109/ICDM.2005.79], which is currently considered one of the best approaches to formalize and discover subsequence anomalies. Discord is defined as a given-length subsequence that is maximally far away from its non-overlapping nearest neighbor. PADDi employs our earlier developed parallel algorithms PD3 [https://doi.org/10.1134/S1054661823020062] and PALMAD [https://doi.org/10.3390/math11143193], which accelerate, respectively, DRAG [https://doi.org/10.1109/ICDM.2007.61] and MERLIN [https://doi.org/10.1109/ICDM50108.2020.00147], on a GPU. The DRAG and MERLIN implements, respectively, fixed- and arbitrary-length discord discovery in a time series stored on disk, proposed by Keogh et al. The PADDi exploits two-level parallelism: first, when the time series is divided into equal-length fragments stored on disks associated with the cluster nodes, and second, when a fragment is split into equal-length segments to be processed by GPUs of the respective node. To implement data exchanges between nodes and calculations on GPUs within a node, we employ MPI and CUDA technologies, respectively. The algorithm performs as follows. Firstly, in each segment processed by one GPU, the algorithm selects potential discords and then discards false positives, resulting in the local candidate set. Next, local candidate sets are sent among cluster nodes in an "all-to-all" manner, resulting in a global candidate set. Then, each cluster node refines the global candidates within its fragment, obtaining the local resulting set of true positive discords. Finally, each cluster node sends the local resulting sets to a master node, which outputs the end result as the intersection of the received local resulting sets.

# Citation
```
@article{KraevaZymbler2025,
 author    = {Yana Kraeva and Mikhail Zymbler},
 title     = {PADDi: Highly Scalable Parallel Algorithm for Discord Discovery on Multi-GPU Clusters},
 journal   = {Lobachevskii Journal of Mathematics},
 volume    = {46},
 number    = {1},
 year      = {2025},
 note      = {(in press)}
}
```
# Acknowledgement
This work was financially supported by the Russian Science Foundation (grant no. 23-21-00465).
