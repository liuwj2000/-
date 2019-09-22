using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp2
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            //Random rnd = new Random();
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(a + b);
            //this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
            MessageBox.Show(textBox1.Text+"和"+ textBox2.Text+"的和是"+textBox3.Text);
            
        }

        private void Button2_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            //Random rnd = new Random();
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(a - b);
           //this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
            MessageBox.Show(textBox1.Text + "和" + textBox2.Text + "的差是" + textBox3.Text);
        }

        private void Button3_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            //Random rnd = new Random();
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(a * b);
            //this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
            MessageBox.Show(textBox1.Text + "和" + textBox2.Text + "的乘积是" + textBox3.Text);
        }

        private void Button6_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            //Random rnd = new Random();
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(a / b);
            //this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
            MessageBox.Show(textBox1.Text + "和" + textBox2.Text + "的向下取整商是" + textBox3.Text);
        }

        private void Button4_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            //Random rnd = new Random();
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(a % b);
            //this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
            MessageBox.Show(textBox1.Text + "模" + textBox2.Text + "的结果是" + textBox3.Text);
        }

        private void Button5_Click(object sender, EventArgs e)
        {
            this.Text = DateTime.Now + "简易计算器——by 刘帅1600012727";
            int a = Convert.ToInt32(textBox1.Text);
            int b = Convert.ToInt32(textBox2.Text);
            textBox3.Text = Convert.ToString(Math.Pow(a,b));     
            MessageBox.Show(textBox1.Text + "的" + textBox2.Text + "次方是" + textBox3.Text);
        }

        private void TextBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void TextBox3_TextChanged(object sender, EventArgs e)
        {

        }

        private void Timer1_Tick(object sender, EventArgs e)
        {
            Random rnd = new Random();
            this.BackColor = Color.FromArgb(rnd.Next(255), rnd.Next(255), rnd.Next(255));
        }
    }
}
