using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace C_SHarp_helloworld
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            MessageBox.Show("hello world");
        }

        private void Button2_Click(object sender, EventArgs e)
        {
            this.Text = "12345";
            //修改状态栏的实验
            this.BackColor = Color.FromArgb(255, 255, 0);
            //修改颜色
        }

        private void Form1_MouseMove(object sender, MouseEventArgs e)
        {
            this.label1.Text = e.X + "," + e.Y;
        }

        private void TextBox1_TextChanged(object sender, EventArgs e)
        {
            textBox2.Text = textBox1.Text;
        }

        private void Timer1_Tick(object sender, EventArgs e)
        {
            Random rnd = new Random();
            this.label1.Left += 10;
            this.Text = DateTime.Now.ToString();
            this.label1.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
        }
    }
}
