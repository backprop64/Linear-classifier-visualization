
let mnist;
let X = [];
let y_classes = [];

let lossHistory = []
for(let i = 0; i < 100; i++){
    lossHistory[i] = 0.0
}

function preload() {
  mnist = loadTable('mnist_test.csv', 'csv', 'header');
}

function setup() {
    for(let r = 0; r <mnist.getRowCount();r++){
        let row = mnist.getRow(r);
        X[r] = []
        y_classes[r] = int(row.getString(0))
        for (let c = 0; c < mnist.getColumnCount()-1; c++) {
            X[r][c] = int(row.getString(c+1))
        }
    }
    createCanvas(600, 380);
    textSize(10);
    // batches
    fill(color('#FFFFE3'));
    noStroke();
    rect(0, 0, 100, 380);
    // templates
    fill(color('#D8F6D6'));
    noStroke();
    rect(100, 0, 100, 380);
    // Weights T
    fill(color('#EEFBDD'));
    noStroke();
    rect(200, 0, 100, 380);
    // Weights T
    fill(color('#C4E8DF'));
    noStroke();
    rect(300, 0, 100, 380);
     // Weights T
     fill(color('#FFE4DD'));
     noStroke();
     rect(400, 0, 100, 380);
      // Weights T
    fill(color('#FFEEDD'));
    noStroke();
    rect(500, 0, 100, 380);

    fill(0);
    text('Class templates', 115, 15);
    text('Classify Digit 0 - 9', 10, 15);

    text('Hyper Parameters', 8, 150);
    text('Activations', 223, 15);
    text('Class Scores', 315, 15);
    text('Current Batch', 420, 15);
    text('Batch Gradient', 515, 15);
    W = InitilizeWeights(10,784)
}

function draw(){
    let BatchIndex = [];
    let dW = zeros(10,784)

    for(let i = 0; i < 100; i++){
        BatchIndex[i] = int(random(0,9500))
    }

    displayBatch(BatchIndex);
    displayWeights(W);


    for(let l = 0; l < 100; l++){
        gradW = softmaxLoss(W,BatchIndex[l],BatchIndex[l]);
        for(let i = 0; i < 10; i++){
            for(let j = 0; j < X[2].length; j++){
                dW[i][j] += gradW[i][j]
            }
        }
    }    
    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            dW[i][j] /= 100
            dW[i][j] += (W[i][j]*2*.0001)//----------------------------reg
        }
    }//reg and average dw 
    displayGrad(dW);
    W = updateWeights(W,dW,.3)//-------------------------------------lr
    displayActivations(W,BatchIndex[0]);
    var currentLoss = 0.0
    for(let i = 0; i < 100; i++){
        currentLoss += lossHistory[i+lossHistory.length-100] 
    }    
    currentLoss/=100

    fill(0);
    rect(18, 100, 60, 18);
    fill(255);
    text('loss: '+ str(currentLoss.toFixed(2)), 25, 113);

    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            dW[i][j] = 0.0
        }
    }

}
function zeros(n,m){
    Weights = []
    for (var i = 0; i < n; i++) {
        Weights[i] = []
        for(var j = 0; j < m; j++){
        Weights[i][j] = 0
        }
    }
    return Weights;
}

function InitilizeWeights(n,m){
    Weights = []
    for (var i = 0; i < n; i++) {
        Weights[i] = []
        for(var j = 0; j < m; j++){
        Weights[i][j] = Math.random();
        }
    }
    return Weights;
}

function displayWeights(Weights){
    let maxval = 0.0
    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            if(Weights[i][j] > maxval){
                maxval = Weights[i][j]
            }
        }
    }
    idx = 0
    for(i = 25; i < 370;i+=35){
        let row = 0;
        for(let j = 0; j < X[2].length; j++){
            if(j%28==0){
                row+=1;
            }
            fill(((Weights[idx][j]*255)/maxval));
            x = j%28
            y= row
            noStroke();
            rect(x+135, y+i , 1, 1);
        }
        idx+=1
    }
}

function displayGrad(Weights){
    let maxval = 0.0
    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            if(Weights[i][j] > maxval){
                maxval = Weights[i][j]
            }
        }
    }
    idx = 0
    for(i = 25; i < 370;i+=35){
        let row = 0;
        for(let j = 0; j < X[2].length; j++){
            if(j%28==0){
                row+=1;
            }
            fill(((Weights[idx][j]*255)/maxval));
            x = j%28
            y= row
            noStroke();
            rect(x+535, y+i , 1, 1);
        }
        idx+=1
    }
}


function displayBatch(arridx){
    idx = 0
    for(i = 25; i < 370;i+=35){
        let row = 0;
        for(let j = 0; j < X[2].length; j++){
            if(j%28==0){
                row+=1;
            }
            fill(X[arridx[idx]][j]);
            x = j%28
            y= row
            noStroke();
            rect(x+434, y+i , 1, 1);
        }
        idx+=1
    }
}

function displayActivations(Weights,index){
    let logits = [];
    for(let i = 0; i < 10; i++){
        logits[i] = 0.0
        for(let j = 0; j < X[2].length; j++){
            logits[i] +=  ((X[index][j]/255)*Weights[i][j])
        }
    }
    maxVal = 0.0
    for(let i = 0; i < 10; i++){
        if(logits[i] > maxVal){
            maxVal=logits[i]
        }
    }
    for(let i = 0; i < 10; i++){
        sum += exp(logits[i]-maxVal)
    }
    for(let i = 0; i < 10; i++){
        logits[i] = -1*log(exp(logits[i] - maxVal)/sum)
      
        fill(logits[i]*35);
        noStroke();
        rect(334, (i*28)+50, 28, 28);
        if(y_classes[index] == i){
            fill("green");
            //noStroke();
            rect(334, (i*28)+50, 10, 10);
        }
        fill(0);
        text(str(i), 320, (i*28)+70);

    }
    

    let maxval = 0.0
    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            if(Weights[i][j] > maxval){
                maxval = Weights[i][j]
            }
        }
    }
    let row = 0
    for(let j = 0; j < X[2].length; j++){
        if(j%28==0){
            row+=1;
        }
        fill(X[index][j]);
        x = j%28
        y= row
        noStroke();
        rect((x*2)+20, (y*2)+25 , 2, 2);
    }

    idx = 0
    for(i = 25; i < 370;i+=35){
        let row = 0;
        for(let j = 0; j < X[2].length; j++){
            if(j%28==0){
                row+=1;
            }
            fill(X[index][j]*(Weights[idx][j]/maxval*2));
            x = j%28
            y= row
            noStroke();
            rect(x+235, y+i , 1, 1);
        }
        idx+=1
    }
}

function softmaxLoss(Weights,index,yval){
    let gradW = zeros(10,784)
    let logits = [];
    for(let i = 0; i < 10; i++){
        logits[i] = 0.0
        for(let j = 0; j < X[2].length; j++){
            logits[i] +=  ((X[index][j]/255)*Weights[i][j])
        }
    }
    sum = 0.0
    maxVal = 0.0
    loss = 0.0
    for(let i = 0; i < 10; i++){
        if(logits[i] > maxVal){
            maxVal=logits[i]
        }
    }
    for(let i = 0; i < 10; i++){
        sum += exp(logits[i]-maxVal)
    }
    for(let i = 0; i < 10; i++){
        logits[i] = (exp(logits[i] - maxVal)/sum)
    }
    for(let j = 0; j < X[2].length; j++){
        gradW[y_classes[index]][j] -= (X[index][j]/255)*(logits[y_classes[index]]-1)
    }///correct class
    let test = 0;
    for(let i = 0; i < 10; i++){
        test += logits[i] 
    }

    for(let i = 0; i < 10; i++){
        if(i != y_classes[index]){
            for(let j = 0; j < X[2].length; j++){
                gradW[y_classes[index]][j] += (X[index][j]/255)*(logits[i])
            }
        }
    }//incorrect classes 
    loss = -1*log(logits[y_classes[index]])
    lossHistory.push(loss)
    console.log(loss)
    return gradW
}


function updateWeights(W,gradW,lr){

    for(let i = 0; i < 10; i++){
        for(let j = 0; j < X[2].length; j++){
            W[i][j] += (gradW[i][j]*lr)
        }
    }
    return W
}

