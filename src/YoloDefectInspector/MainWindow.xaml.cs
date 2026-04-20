using System.Windows;
using YoloDefectInspector.ViewModels;

namespace YoloDefectInspector;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel();
    }
}
