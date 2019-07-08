"use strict";
//TODO:
// 1. 锚定，用户绘图时，如果很接近某个交点、某条直线、某个圆周等，释放鼠标时，点落的位置应该是最接近的交点或线上。
// 2. 线附近的点？点所在的线应该高亮？
// 3. 相同的位置画2个点，2条线，应去重？
// 4. 超出可视区域部分的处理? 画步可移动、缩放？
// 5. 配色 

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
            POINT: 1, //点
            LINE: 2, //直线
            CIRCLE: 3 //圆
        };
        if (Object.freeze) {
            Object.freeze(Tools);
        }

        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var canvas_rect = canvas.getBoundingClientRect();

        //定义各种变量
        var current_tool = Tools.HAND; //当前用户选择的绘图工具，默认移动  
        var operation_list = []; //用户的绘图记录 
        //{'x':1,'y':2,'items':[[1,3],[5,7]]} 
        var manual_points = [];  //手工绘制的点
        var intersect_points = []; //相交点
        var all_points = [];  //人工绘制的所有点，以及各种线条相交点

        var isDrawing = false;
        var requestAnimationFrame_status = 0;

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
            // y = a0*x + b0 , 将x代入公式得到y 
            var line = calc_line_parameters(item);
            var line1 = calc_line_parameters(item1);
            var x = (line1.bias - line.bias) / (line.weight - line1.weight);
            var y = line.weight * x + line.bias;

            if (x <= 0 || y <= 0 || x > canvas_rect.width || y > canvas_rect.height) {
                //TODO: 画布平移、缩放？
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
            var radius = calc_distance(circle.x, circle.y, circle.x1, circle.y1);

            var A = Math.pow(line.weight, 2) + 1;
            var B = line.bias - circle.y;
            var C = (B * line.weight - circle.x) / A
            var t = (Math.pow(radius, 2) - Math.pow(circle.x, 2) - Math.pow(B, 2)) / (Math.pow(line.weight, 2) + 1)
            var f = t + Math.pow(C, 2);
            if (f < 0) {
                //不存在交点
                //工程上应该允许有个误差值？误差多少合适？相切的点？
                // 过圆心、垂直与直线的线段，线段的距离是否等于半径？
                return false;
            }
            var l = Math.sqrt(f)

            var x1 = l - C;
            var y1 = line.weight * x1 + line.bias;
            var x2 = - l - C;
            var y2 = line.weight * x2 + line.bias;

            return [{ 'x': Math.round(x1), 'y': Math.round(y1) },
            { 'x': Math.round(x2), 'y': Math.round(y2) }];
        }

        function calc_circle2_intersect_point(circle, circle1) {
            //计算圆和圆的相交点
            //圆方程： (x-x0)²+(y-y0)²=r0²
            //  (x-x1)²+(y-y1)²=r1²

        }

        function calc_chuizhidian(line_item, circle) {
            return false;
            //TODO:
            //过圆心且垂直与线段的直线，与
            //计算直线上、距离圆心最近的点, 可以用来判断圆的切线
            //设该点为x,y
            //圆心-线段端点 距离
            // a² + b² = c²
            // c² = 
            var line = calc_line_parameters(line_item);
            var radius = calc_distance(circle.x, circle.y, circle.x1, circle.y1);

            var cc = Math.pow(circle.x - line_item.x, 2) + Math.pow(circle.y - line_item.y, 2);
            var A = 2 * (Math.pow(line.weight, 2) + 1);
            var B = (2 * line.bias - circle.x - circle.y - line_item.x - line_item.y) / A;
            var C = Math.pow(circle.x, 2) + Math.pow(line.bias - circle.y, 2) + Math.pow(line_item.x, 2) + Math.pow(line.bias - line_item.y, 2);

            var D = (cc - C) / A;
            var E = (cc - C) / A + Math.pow(B, 2);

            // console.log(D);

            if (E < 0) {
                //不存在交点
                //工程上应该允许有个误差值？误差多少合适？相切的点？
                // 过圆心、垂直与直线的线段，线段的距离是否等于半径？
                return false;
            }
            var l = Math.sqrt(E)

            var x1 = l - B
            var y1 = line.weight * x1 + line.bias;

            var x2 = -l - B
            var y2 = line.weight * x1 + line.bias;

            var ret = [{ 'x': Math.round(x1), 'y': Math.round(y1) },
            { 'x': Math.round(x2), 'y': Math.round(y2) }];

            return ret;

        }

        function get_nearest_point(x, y, points) {
            //计算给定一点与所有相交点之间的距离
            const DUP_POINT_DISTANCE_THRESHOLD = 10;

            var dist_list = []
            for (var item of points) {
                var d = Math.sqrt((x - item.x) * (x - item.x) + (y - item.y) * (y - item.y));
                dist_list.push(d);
            }

            var min_d = Math.min.apply(null, dist_list);
            if (min_d > DUP_POINT_DISTANCE_THRESHOLD) {
                return false;
            }

            for (var i = 0; i < dist_list.length; i++) {
                if (dist_list[i] == min_d) {
                    return points[i];
                }
            }

            return false;
        }

        function get_intersect_points() {
            //获取所有的相交点
            var intersect_points = [];

            //计算两个图形相交的点
            var len = operation_list.length;
            if (len < 2) {
                return intersect_points;
            }
            for (var i = 0; i < len; i++) {
                for (var j = i + 1; j < len; j++) {
                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.LINE) {
                        //两直线相交
                        var points = calc_line2_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            //是否需要在这里去重？
                            intersect_points.push.apply(intersect_points, points);
                        }
                    }

                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.CIRCLE) {
                        //直线&圆
                        var points = calc_line_circle_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            intersect_points.push.apply(intersect_points, points);
                        }

                        var points = calc_chuizhidian(operation_list[i], operation_list[j]);
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
                        var points = calc_chuizhidian(operation_list[j], operation_list[i]);
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
            return intersect_points;
        }


        function get_manual_points() {
            //人工绘制的点
            var tmp = [];
            var len_operation_list = operation_list.length;
            for (var i = 0; i < len_operation_list; i++) {
                var item = operation_list[i];

                //点 
                if (item.type == Tools.POINT) {
                    tmp.push({ 'x': item.x, 'y': item.y });
                }

                //线 or 圆
                if (item.type == Tools.LINE || item.type == Tools.CIRCLE) {
                    tmp.push({ 'x': item.x, 'y': item.y });
                    tmp.push({ 'x': item.x1, 'y': item.y1 });
                }
            }

            //去重？
            var manual_points = [];
            for (var i = 0; i < tmp.length; i++) {
                //TODO: 判断是否为最后一个 ？

                var d = get_nearest_point(tmp[i].x, tmp[i].y, manual_points);
                if (!d) {
                    // console.log(tmp[i]);
                    manual_points.push({ 'x': tmp[i].x, 'y': tmp[i].y });
                }
            }

            return manual_points;
        }



        function process_points() {
            //人工point：点、线段的起始点、圆心(起点)和圆周(终点)等  
            var last_index = operation_list.length - 1;
            if (last_index < 0) {
                return false;
            }
            // var last_item = operation_list[last_index];

            var last_item = operation_list.pop();

            all_points = [];
            all_points.push.apply(all_points, get_manual_points());
            all_points.push.apply(all_points, get_intersect_points());

            // 判断起始点
            var nearest_point = get_nearest_point(last_item.x, last_item.y, all_points);
            if (nearest_point) {
                last_item.x = nearest_point.x;
                last_item.y = nearest_point.y;
            }

            // 判断终点
            if (last_item.type == Tools.LINE || last_item.type == Tools.CIRCLE) {
                var nearest_point = get_nearest_point(last_item.x1, last_item.y1, all_points);
                if (nearest_point) {
                    last_item.x1 = nearest_point.x;
                    last_item.y1 = nearest_point.y;
                }

                if (last_item.type == Tools.CIRCLE) {
                    var radius = calc_distance(last_item.x, last_item.y, last_item.x1, last_item.y1);
                    last_item.radius = radius;
                }
            }

            //线与线相交的点，如果与人工的接近，按人工的为准？
            operation_list.push(last_item);

            // all_points = [];
            // all_points.push.apply(all_points, get_manual_points());  
            // all_points.push.apply(all_points, get_intersect_points());

        }

        function render() {
            //绘制图形
            if (current_tool == Tools.HAND) {
                //TODO ?
                return;
            }

            canvas_rect = canvas.getBoundingClientRect();
            ctx.clearRect(0, 0, canvas_rect.width, canvas_rect.height); //TODO 

            //计算所有点：人工点和交点
            process_points();

            //绘制人工点
            var manual_points = get_manual_points();
            for (var item of manual_points) {
                ctx.beginPath();
                ctx.strokeStyle = 'black';
                ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.stroke();
            }

            //绘制交点
            var intersect_points = get_intersect_points();
            for (var item of intersect_points) {
                ctx.beginPath();
                ctx.strokeStyle = 'gray';
                ctx.arc(item.x, item.y, 2, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.stroke();
            }

            // js for 遍历数组
            //https://juejin.im/post/5a3a59e7518825698e72376b
            for (var item of operation_list) {
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
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 1;
                    ctx.moveTo(item.x, item.y);
                    ctx.lineTo(item.x1, item.y1);
                    ctx.stroke();
                    ctx.closePath();
                }

                //绘制圆
                if (item.type == Tools.CIRCLE) {
                    ctx.beginPath();
                    ctx.strokeStyle = 'black';
                    ctx.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
                    ctx.closePath();
                    ctx.stroke();
                }
            }

        }

        //事件绑定
        //https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event  
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
            canvas.addEventListener('mousedown', function (e) {
                if (current_tool == Tools.HAND) {
                    return;
                }
                isDrawing = true;
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;
                var obj = { "type": current_tool, 'x': x, 'y': y, 'x1': x, 'y': y };
                operation_list.push(obj);
                requestAnimationFrame_status = window.requestAnimationFrame(render);
            });
            canvas.addEventListener('mousemove', function (e) {
                if (!isDrawing) {
                    return false;
                }

                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return false;
                }

                operation_list[last_index].x1 = e.clientX - canvas_rect.left;
                operation_list[last_index].y1 = e.clientY - canvas_rect.top;

                requestAnimationFrame_status = window.requestAnimationFrame(render);

            });
            canvas.addEventListener('mouseup', function (e) {
                if (!isDrawing) {
                    return false;
                }

                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return false;
                }

                //如果两点的距离<n,则认为是无效的操作？ 
                isDrawing = false;
                window.cancelAnimationFrame(requestAnimationFrame_status);
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
            });

            window.addEventListener('load', function (e) {
                //页面加载，load用户的历史操作记录，呈现给用户？
            });
        }


        event_bindding();
    }

    main();

})();