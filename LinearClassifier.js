//Kinda ugly setup make model object oriented 

var Weights = [];

var fs = require('file-system');

var dataFileBuffer  = fs.readFileSync(__dirname + '/train-images-idx3-ubyte');
var labelFileBuffer = fs.readFileSync(__dirname + '/train-labels-idx1-ubyte');
var Mnist = [];


// It would be nice with a checker instead of a hard coded 60000 limit here
for (var image = 0; image < 100; image++) { 
    var pixels = [];

    for (var x = 0; x <= 27; x++) {
        for (var y = 0; y <= 27; y++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
        }
    }
    var imageData  = [];
    imageData[0] = Number(JSON.stringify(labelFileBuffer[image + 8]))
    imageData[1] = pixels;
    Mnist.push(imageData);
}

//////////////////////////////////////////////////////
// basic implementation of simple LinAlg operations //
//////////////////////////////////////////////////////

function sum(v,w){
  var sum = [];
  for(var i = 0; i < v.length; i++){
    sum[i] = v[i] + w[i]
  }
  return sum
}
function sumTotal(v){
  var sum = 0.0;
  for(var i = 0; i < v.length; i++){
    sum += v[i]
  }
  return sum
}

function dot(v,w){
  var dotProduct = 0;
  for(var i = 0; i < v.length; i++){
    dotProduct += (v[i]*w[i])
  }
  return dotProduct
}

function scale(v,c){
  for(var i = 0; i < v.length; i++){
    v[i] = c*v[i]
  }
  return v
}

function exp(v){
  for(var i = 0; i< v.length; i++){
    v[i] = 2.71**v[i]
  }
  return v 
}
function normalize(v){
  max = v[0]
  for(var i = 0; i< v.length; i++){
    if(v[i] > max){
      max = v[i]
    }
  }
  for(var i = 0; i< v.length; i++){
    v[i] -= max
  }
  return v 
}

function transpose(A){
  var At = [];
  for(var i = 0; i < A.length; i++){
    At[i] = []
    for(var j = 0; j < A[0].length; j++){
      At[i][j] = A[j][i]
    }
  }
  return(At)
}

/////////////////////////////////////
///initialization and forward pass///
/////////////////////////////////////

function InitilizeWeights(Weights){
    for (var i = 0; i < 10; i++) {
      Weights[i] = []
      for(var j = 0; j < (28*28); j++){
        Weights[i][j] = Math.random();
      }
    }
    return Weights;
}
function Zeros(n,m){
  Weights = []
  for (var i = 0; i < n; i++) {
    Weights[i] = []
    for(var j = 0; j < m; j++){
      Weights[i][j] = 0;
    }
  }
  return Weights;
}

////////////////////
///Loss Functions///
////////////////////


function SVMloss(Weights,data){
  num_classes = Weights.length
  num_train = data.length
  dW = Zeros(10,784)
  loss = 0

  for(var i = 0; i < num_train; i++){
    var logits = [];
    for(var score = 0; score< num_classes; score++){
      logits[score] = dot(Weights[score],data[i][1])
    }

    correctclass = logits[data[i][0]]

    for(var j = 0; j < num_train; j++){
      if(j != data[i][0]){
        margin = (logits[j] - correctclass + 1) 
        if(margin > 0){
          loss += margin
          dW[data[i][0]] -= data[i][1]
          dW[j] += data[i][1]
        }
      }
      
    }
  }
  loss /= num_train

  for(var i = 0; i < 10; i++) {
    for(var j = 0; j < 784; j++){
      dW[i][j] /= num_train;
    }
  }
  return loss, dW
}


////////////////////
/// Optimization ///
////////////////////

function updateWeights(Weights,dW,lr){
  for(var i = 0; i < 10; i++) {
    for(var j = 0; j < 784; j++){
      Weights[i][j] -= (dW[i][j]*lr)
    }
  }
  return Weights
}

function updateCanvas(Weights){

  var c = document.getElementById("template0");
  var ctx = c.getContext("2d");
  var imgData = ctx.createImageData(28, 28);
  var i;
  
  for (i = 0; i < imgData.data.length; i += 4){
    imgData.data[i + 0] = 255*Weights[0][i];
    imgData.data[i + 1] = 255*Weights[0][i];
    imgData.data[i + 2] = 255*Weights[0][i];
    imgData.data[i + 3] = 255*Weights[0][i];
  }
  ctx.putImageData(imgData, 10, 10);
}

var loss = 0.0

Weights = InitilizeWeights(Weights)
updateCanvas(Weights)

for(var j = 0; j < 10; j++){
  loss, dW = SVMloss(Weights,Mnist)
  Weights = updateWeights(Weights,dW,.003)
  console.log(loss)
}



