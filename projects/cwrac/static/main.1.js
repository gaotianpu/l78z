function draw() {


    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    // if (!canvas.getContext) {
    //     return;
    // }


    // 让canvas宽度占满整个浏览器内可视区域？ 
    // https://www.cnblogs.com/polk6/p/5051935.html
    // body.clientHeight;
    // body.clientWidth;

    function set_canvas_size() {
        //设置为全屏幕
        canvas.width = document.documentElement.clientHeight;
        canvas.height = document.documentElement.clientWidth;
        console.log(document.documentElement.clientHeight, document.documentElement.clientWidth);
    }

    set_canvas_size();
    window.onresize = function () {
        set_canvas_size();
    }




    

    function draw_line(x0, y0, x1, y1) {
        // 直线: x0,y0,x1,y1,端点的样式，颜色，透明度，宽度，

        //https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial/Applying_styles_and_colors
        ctx.lineWidth = 1;
        // ctx.lineCap
        // ctx.lineJoin
        // ctx.miterLimit
        // ctx.getLineDash()
        // ctx.setLineDash(segments)
        // ctx.lineDashOffset = value


        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        // ctx.fill(); 
        // ctx.arc(x0, y0, 3, 0, Math.PI * 2, true);
        // ctx.arc(x1, y1, 3, 0, Math.PI * 2, true);
        // ctx.fill(); 
        // ctx.closePath();
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x0, y0, 3, 0, Math.PI * 2, true);
        ctx.arc(x1, y1, 3, 0, Math.PI * 2, true);
        // ctx.closePath();
        ctx.fill();
    }

    function draw_circle(x0, y0, radius) {
        //http://www.w3school.com.cn/tags/canvas_arc.asp
        ctx.strokeStyle = 'green';

        ctx.lineWidth = 5;

        ctx.beginPath();
        // ctx.moveTo(x0,y0);
        // arc(x, y, radius, startAngle, endAngle, anticlockwise)
        // 以（x,y）为圆心的以radius为半径的圆弧（圆），从startAngle开始到endAngle结束，按照anticlockwise给定的方向（默认为顺时针）来生成。 

        ctx.arc(x0, y0, radius, 0, Math.PI * 2, true);
        // ctx.moveTo(x0,y0);
        ctx.closePath();
        // ctx.moveTo(x0,y0);
        ctx.stroke();
    }


    // ctx.fillStyle = 'green'; //颜色，透明度
    // ctx.fillStyle = "rgb(200,0,0)";
    // ctx.fillStyle = "rgba(0, 0, 200, 0.5)";
    // ctx.fillRect(10, 10, 100, 100);

    // 光线投影(ray-casting),
    // https://hacks.mozilla.org/2013/05/optimizing-your-javascript-game-for-firefox-os/

    //https://developer.mozilla.org/en-US/docs/Games/Techniques/Efficient_animation_for_web_games

    //https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial/Advanced_animations

    canvas.addEventListener('mouseout', function (e) {
        // window.cancelAnimationFrame(raf);
        // running = false;
    });

    canvas.addEventListener('mousemove', function (e) {
        // if (!running) {
        //     // clear();
        //     // ball.x = e.clientX;
        //     // ball.y = e.clientY;
        //     // ball.draw();
        // }
    });

    canvas.addEventListener('click', function (e) {
        // if (!running) {
        //     // raf = window.requestAnimationFrame(draw);
        //     // running = true;
        // }
    });

     






    draw_circle(50, 50, 20)
    draw_line(20, 20, 80, 80)


    var rectangle = new Path2D();
    rectangle.rect(10, 10, 50, 50);

    var circle = new Path2D();
    circle.moveTo(125, 35);
    circle.arc(100, 35, 25, 0, 2 * Math.PI);

    ctx.stroke(rectangle);
    ctx.fill(circle);


    //数据结构
    // 点: x,y,半径,颜色，透明度

    // 圆：x0,y0,半径，颜色，透明度
}
