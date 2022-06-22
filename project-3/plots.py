import matplotlib.pyplot as plt

path_all_metrics_plots = "./images/all_metrics/"

def plot_sensor_data(new_PR_data_inner):
    # PLOT raw data from Sensor_O3 against date
    new_PR_data_inner.plot(x='date', y='Sensor_O3')
    plt.savefig(path_all_metrics_plots + "sensor-o3_date.png")
    #plt.show()
    plt.clf()

    # PLOT Refst from expensive Sensor_O3 against date
    new_PR_data_inner.plot(x='date', y='RefSt', color='red')
    plt.savefig(path_all_metrics_plots + "refst_date.png")
    #plt.show()
    plt.clf()

    # SCATTER PLOT LOW COST SENSOR O3 AGAINST REFST
    new_PR_data_inner.plot.scatter(x='Sensor_O3', y='RefSt', color='green')
    plt.savefig(path_all_metrics_plots + "sensor-o3_refst.png")
    #plt.show()
    plt.clf()

    # Normalize the data
    new_PR_data_inner['Sensor_O3_norm'] = (new_PR_data_inner['Sensor_O3'] - new_PR_data_inner['Sensor_O3'].mean()) / \
                                          new_PR_data_inner['Sensor_O3'].std()
    new_PR_data_inner['RefSt_norm'] = (new_PR_data_inner['RefSt'] - new_PR_data_inner['RefSt'].mean()) / \
                                      new_PR_data_inner[
                                          'RefSt'].std()

    # SCATTER PLOT LOW COST SENSOR O3 AGAINST REFST but now normalized data
    normalized_plt = new_PR_data_inner.plot.scatter(x='Sensor_O3_norm', y='RefSt_norm', color='green')
    normalized_plt.set_xlabel("Sensor_O3 normalized")
    normalized_plt.set_ylabel("RefSt normalized")
    plt.savefig(path_all_metrics_plots + "norm_sensor-o3_refst.png")
    #plt.show()
    plt.clf()

    columns_plot = new_PR_data_inner.columns[3:-2]  # Select only necessary columns

    # PLOTS O3 against all metrics and RefSt against all metrics
    for i in columns_plot:
        new_PR_data_inner.plot.scatter(x='Sensor_O3', y=i)
        #plt.show()
        plt.savefig(path_all_metrics_plots + "sensor-o3_" + i + ".png")
        plt.clf()

    for i in columns_plot:
        new_PR_data_inner.plot.scatter(x='RefSt', y=i, color='red')
        plt.savefig(path_all_metrics_plots + "refst_" + i + ".png")
        #plt.show()
        plt.clf()
