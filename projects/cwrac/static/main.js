"use strict";
//TODO:
// 1. 锚定，用户绘图时，如果很接近某个交点、某条直线、某个圆周等，释放鼠标时，点落的位置应该是最接近的交点或线上。
// 2. 超出可视区域部分的处理? 画步可移动、缩放？
// 

// 
//1. 工具栏
// 基础工具：移动、点、直线、圆
// 高级工具：需逐步解锁，垂直平分线、垂线、二分角、平行线、
// 当前选中的是什么工具? ！ 解锁工具
//2. 不同的工具对鼠标事件响应不一样？鼠标的点击、释放

//1. 有一个数组，用于记录每个步骤干嘛
//2. 事件响应，鼠标的点击、释放
; (function () {
    function main() {
        //工具箱，枚举值
        var Tools = {
            HAND: 0, //0-移动
            POINT: 1, //
            LINE: 2, //
            CIRCLE: 3
        };
        if (Object.freeze) {
            Object.freeze(Tools);
        }

        var current_tool = Tools.HAND; //默认 

        var operation_list = []; //用户的绘图记录

        //各种线条相交点
        //{'x':1,'y':2,'items':[[1,3],[5,7]]}
        var intersect_points = [];

        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var canvas_rect = canvas.getBoundingClientRect();

        //各种数学计算公式 

        function calc_distance(x, y, x1, y1) {
            //计算两点之间的距离
            return Math.round(Math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1)));
        }

        function calc_line_parameters(item) {
            //计算直线的斜率weight、偏置量bias、直线在画布左右边界的交点
            //y = w*x + b 
            var w = (item.y - item.y1) / (item.x - item.x1); //直线的斜率
            var b = item.y - w * item.x;  //直线的偏置项

            //直线在画布左右的边界点
            var left = { 'x': 0, 'y': Math.round(b) };
            var right = { 'x': canvas_rect.width, 'y': Math.round(w * canvas_rect.width + b) };

            return { 'weight': w, 'bias': b, 'left': left, 'right': right }
        }

        function calc_line2_intersect_point(item, item1) {
            //计算两条直线的交点
            // y = a0*x + b0
            // y1 = a1*x1 + b1 , 相交点，x=x1, y=y1
            // x = (b1-b0)/(a0-a1), 将x代入任一直线方程求解
            // y= a0*x + b0 , 将x代入公式得到y 
            var line = calc_line_parameters(item);
            var line1 = calc_line_parameters(item1);
            var x = (line1.bias - line.bias) / (line.weight - line1.weight);
            var y = line.weight * x + line.bias;

            if (x <= 0 || y <= 0 || x > canvas_rect.width || y > canvas_rect.height) {
                //TODO: 画布移动？
                return false;
            }
            return [{ 'x': Math.round(x), 'y': Math.round(y) }];
        }

        function calc_line_circle_intersect_point(line_item, circle) {
            //计算线段和圆的交点
            //https://thecodeway.com/blog/?p=932 
            // 直线方程： y = ax + b 
            // 圆方程： (x-x0)²+(y-y0)²=r²  //圆心坐标为(x0，y0),r半径 
            // 推导：
            // (x-x0)² + (ax+b-y0)²=r² //相交点x,y坐标相等，联立方程，求x值  
            // (x²+x0²-2*x0*x) + (a²x² + (b-y0)² + 2*a*x*(b-y0) )= r²
            // (x² + a²x²) + (2*a*(b-y0)*x-2*x0*x) + (x0² + (b-y0)²) = r²
            // (a²+1)x² + 2*(a*(b-y0)-x0)*x + (x0² + (b-y0)²) = r²
            // (a²+1)x² + 2*(a*(b-y0) - x0) * x = r² - (b-y0)² - x0²
            // x² + 2*((a*(b-y0) - x0)/(a²+1)) * x = (r² - (b-y0)² - x0²)/(a²+1)
            // x² + 2*((a*(b-y0) - x0)/(a²+1)) * x + ((a*(b-y0) - x0)/(a²+1))² = (r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²
            // (x+ [((a*(b-y0) - x0)/(a²+1))²]开放)² = (r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²
            // x+ [((a*(b-y0) - x0)/(a²+1))²]开放 = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放
            // x = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放 - [((a*(b-y0) - x0)/(a²+1))²]开放
            var line = calc_line_parameters(line_item);
            var a = line.weight;
            var b = line.bias;
            var r = calc_distance(circle.x, circle.y, circle.x1, circle.y1);
            var x0 = circle.x;
            var y0 = circle.y;

            var A = Math.pow(a, 2) + 1;
            var B = b - y0;
            var C = (B * a - x0) / A
            var t = (Math.pow(r, 2) - Math.pow(x0, 2) - Math.pow(B, 2)) / (Math.pow(a, 2) + 1)
            var f = t + Math.pow(C, 2);

            if (f < 0) {
                return false;
            }

            var l = Math.sqrt(f)

            var x1 = l - C;
            var y1 = a * x1 + b;

            var x2 = - l - C;
            var y2 = a * x2 + b;

            // console.log(x1, y1, x2, y2);

            return [{ 'x': Math.round(x1), 'y': Math.round(y1) },
            { 'x': Math.round(x2), 'y': Math.round(y2) }];
        }

        function calc_circle2_intersect_point(circle, circle1) {
            //计算直线和圆相交点
        }

        function get_all_intersect_points() {
            //获取所有的相交点
            intersect_points = [];

            //计算两个图形相交的点
            var len = operation_list.length;
            if (len < 2) {
                return;
            }
            for (var i = 0; i < len; i++) {
                for (var j = i + 1; j < len; j++) {
                    if (i == j) {
                        continue;
                    }

                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.LINE) {
                        //两直线相交
                        var points = calc_line2_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            intersect_points.push.apply(intersect_points, points);
                        }
                    }

                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.CIRCLE) {
                        //直线&圆
                        var points = calc_line_circle_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            intersect_points.push.apply(intersect_points, points);
                        }

                    }

                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.LINE) {
                        //圆&直线
                        var points = calc_line_circle_intersect_point(operation_list[j], operation_list[i]);
                        if (points) {
                            intersect_points.push.apply(intersect_points, points);
                        }

                    }
                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.CIRCLE) {
                        //圆&圆
                        var points = calc_circle2_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            intersect_points.push.apply(intersect_points, points);
                        }
                    }

                    // console.log(operation_list[i], operation_list[j]);
                }

            }

            //点去重，1.相交点去重，2.与用户绘制的点去重
        }


        function render() {
            //绘制图形
            if (current_tool == Tools.HAND) {
                //TODO ?
                return;
            }

            canvas_rect = canvas.getBoundingClientRect();
            ctx.clearRect(0, 0, canvas_rect.width, canvas_rect.height); //TODO

            //绘制各种交点
            get_all_intersect_points();
            for (var item of intersect_points) {
                ctx.beginPath();
                ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.stroke();
            }

            // js for 遍历数组
            //https://juejin.im/post/5a3a59e7518825698e72376b
            for (var item of operation_list) {
                //点
                if (item.type == Tools.POINT) {
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.closePath();
                    ctx.stroke();
                }

                //线
                if (item.type == Tools.LINE) {
                    //绘制直线
                    var line = calc_line_parameters(item);
                    ctx.beginPath();
                    ctx.lineWidth = 0.3;
                    ctx.strokeStyle = 'gray';
                    ctx.moveTo(line.left.x, line.left.y);
                    ctx.lineTo(line.right.x, line.right.y);
                    ctx.stroke();
                    ctx.closePath();

                    //绘制两点之间的线段部分
                    ctx.beginPath();
                    ctx.moveTo(item.x, item.y);
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 1;
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                    ctx.moveTo(item.x, item.y);
                    ctx.lineTo(item.x1, item.y1);
                    ctx.arc(item.x1, item.y1, 2, 0, Math.PI * 2, true);
                    ctx.stroke();
                    ctx.closePath();

                }

                //圆
                if (item.type == Tools.CIRCLE) {
                    //绘制圆心
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, 2, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.arc(item.x1, item.y1, 2, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                    //绘制圆周
                    var radius = calc_distance(item.x, item.y, item.x1, item.y1);
                    ctx.beginPath();
                    ctx.arc(item.x, item.y, radius, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();

                }
            }

        }

        //事件绑定
        // https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event 

        function event_bindding() {
            //绘图工具框
            var toolbox = document.getElementById("toolbox");
            toolbox.addEventListener('click', function () {
                var e = event || window.event;
                if (e.target && e.target.nodeName.toUpperCase() == "INPUT") {
                    current_tool = e.target.value;
                }
            }, false);

            //两种习惯：暂时先支持第一种
            //1. 拖拽式 
            //2. 点击式 1.鼠标点击2下确定一条直线，一个圆等  
            var isDrawing = false;
            var requestAnimationFrame_status = 0;
            canvas.addEventListener('mousedown', function (e) {
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;
                isDrawing = true;

                if (current_tool != Tools.HAND) {
                    var obj = { "type": current_tool, 'x': x, 'y': y, 'x1': x, 'y': y };
                    operation_list.push(obj);
                }

                requestAnimationFrame_status = window.requestAnimationFrame(render);
            });
            canvas.addEventListener('mousemove', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return;
                }

                if (isDrawing === true) {
                    var x = e.clientX - canvas_rect.left;
                    var y = e.clientY - canvas_rect.top;

                    operation_list[last_index].x1 = x;
                    operation_list[last_index].y1 = y;

                    requestAnimationFrame_status = window.requestAnimationFrame(render);
                }
            });
            canvas.addEventListener('mouseup', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return;
                }

                if (isDrawing === true) {
                    isDrawing = false;

                    var x = e.clientX - canvas_rect.left;
                    var y = e.clientY - canvas_rect.top;
                    operation_list[last_index].x1 = x;
                    operation_list[last_index].y1 = y;

                    window.cancelAnimationFrame(requestAnimationFrame_status);
                }
            });

            canvas.addEventListener('click', function (e) {
                //
            });
            //右键单击、取消？

            //浏览器大小调整
            window.addEventListener('resize', function (e) {
                var height = document.documentElement.clientHeight;
                var width = document.documentElement.clientWidth;
                canvas.height = height;
                canvas.width = width;
                requestAnimationFrame_status = window.requestAnimationFrame(render);
                // console.log(height, width);
            });

            window.addEventListener('load', function (e) {
                //页面加载，load用户的历史操作记录，呈现给用户？
            });
        }


        event_bindding();
    }

    main();
})();