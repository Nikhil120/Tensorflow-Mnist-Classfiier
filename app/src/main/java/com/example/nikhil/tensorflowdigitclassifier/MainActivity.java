package com.example.nikhil.tensorflowdigitclassifier;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.icu.text.LocaleDisplayNames;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    ImageView imageView;
    TextView textView;
    private TensorFlowInferenceInterface tensorFlowInferenceInterface;
    private String inputName = "input";
    private String outputName = "output";
    private static List<String> labels;
    private static AssetManager assetManager;
    private Bitmap bitmap;
    float[] output = new float[10];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.imageView);
        textView = (TextView) findViewById(R.id.result);

        assetManager = getAssets();
        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(assetManager, "opt_mnist_convnet.pb");
        try {
            readLabels();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void readLabels() throws IOException {
        labels = new ArrayList<>();

        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(assetManager.open("mnist_labels.txt")));
        String line = "";

        while ((line = bufferedReader.readLine()) != null){
            labels.add(line);
        }

        bufferedReader.close();
    }

    public void onCapture(View view){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, 8888);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == 8888){
            bitmap = (Bitmap) data.getExtras().get("data");
            bitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, false);
            imageView.setImageBitmap(bitmap);
        }
    }

    public void onClassify(View view){
        int[] pixels = new int[784];
        float[] normalizedPixels = new float[784];

        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i=0; i<pixels.length; ++i) {
            int p = pixels[i];
            Log.d("Pixels", String.valueOf(p));
            int b = p & 0xff;
            normalizedPixels[i] = (float) ((0xff - b) / 255.0);
        }

        tensorFlowInferenceInterface.feed(inputName, normalizedPixels, 1, 28, 28, 1);

        tensorFlowInferenceInterface.feed("keep_prob", new float[]{ 1 });

        tensorFlowInferenceInterface.run(new String[]{ outputName });

        tensorFlowInferenceInterface.fetch(outputName, output);

        float max = 0.00f;
        int pos = 0;


        for (int i=0; i < output.length; i++){
            if (output[i] > max){
                max = output[i];
                pos = i;
            }
        }
        textView.setText(labels.get(pos));
    }
}
