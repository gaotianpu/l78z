; "use strict";
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

        //各种配置参数
        const HIGHTLIGHT_COLOR = 'rgb(60,160,250)'; //高亮颜色
        const DASHES_COLOR = 'rgb(60,60,60)'; //虚线颜色
        const NORMAL_COLOR = 'rgb(30,30,30)'; //正常的颜色
        const DUP_POINT_DISTANCE_THRESHOLD = 10; //磁吸距离

        //页面元素
        var toolbox = document.getElementById("toolbox");
        var btn_undo = document.getElementById("btn_undo");
        var btn_redo = document.getElementById("btn_redo");
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var canvas_rect = canvas.getBoundingClientRect();

        //定义各种变量 
        var operation_list = []; //用户的绘图记录
        var undo_operation_list = []; //撤销操作临时保存  
        //{'x':1,'y':2,'items':[[1,3],[5,7]]}  
        var all_points = [];  //人工绘制的所有点以及各种线条相交点

        var current_tool = Tools.HAND; //当前用户选择的绘图工具，默认移动  
        var isDrawing = false;
        var requestAnimationFrame_status = 0;
        var movie_points = { 'x': 0, 'y': 0, 'x_movie': 0, 'y_movie': 0 };

        //各种数学计算公式 

        //计算两点之间的距离 
        function calc_distance(item) {
            var ret = Math.sqrt(Math.pow(item.x - item.x1, 2) + Math.pow(item.y - item.y1, 2));
            return Math.round(ret);
        }

        //计算直线的斜率weight、偏置量bias、直线在画布左右边界的交点
        function calc_line_parameters(item) {
            //y = w*x + b 
            var w = (item.y - item.y1) / (item.x - item.x1); //直线的斜率
            var b = item.y - w * item.x;  //直线的偏置项

            //直线在画布左右的边界点
            var left = { 'x': 0, 'y': Math.round(b) };
            var right = { 'x': canvas_rect.width, 'y': Math.round(w * canvas_rect.width + b) };

            return { 'weight': w, 'bias': b, 'left': left, 'right': right }
        }

        //判断给定point是否在canvas边界内
        function in_boundary(x, y) {
            if (x < 0 || y < 0 || x > canvas_rect.width || y > canvas_rect.height) {
                //TODO: 画布平移、缩放？
                return false;
            }
            return true;
        }

        //计算两条直线的交点
        function calc_line2_intersect_point(line, line1) {
            // y = a0*x + b0
            // y1 = a1*x1 + b1 , 相交点，x=x1, y=y1
            // x = (b1-b0)/(a0-a1), 将x代入任一直线方程求解
            // y = a0*x + b0 , 将x代入公式得到y  
            var x = (line1.bias - line.bias) / (line.weight - line1.weight);
            var y = line.weight * x + line.bias;

            //平行或重合
            if (line1.weight == line.weight) {
                return false;
            }
            if (!in_boundary(x, y)) {
                return false;
            }

            if (x <= 0 || y <= 0 || x > canvas_rect.width || y > canvas_rect.height) {
                //TODO: 画布平移、缩放？
                return false;
            }
            return [{ 'x': Math.round(x), 'y': Math.round(y) }];
        }

        //计算线段和圆的交点
        function calc_line_circle_intersect_points(line, circle) {
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
            // (x+ [((a*(b-y0) - x0)/(a²+1))²]开方)² = (r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²
            // x+ [((a*(b-y0) - x0)/(a²+1))²]开方 = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放
            // x = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放 - [((a*(b-y0) - x0)/(a²+1))²]开放 
            var A = Math.pow(line.weight, 2) + 1;
            var B = line.bias - circle.y;
            var C = (B * line.weight - circle.x) / A
            var t = (Math.pow(circle.radius, 2) - Math.pow(circle.x, 2) - Math.pow(B, 2)) / (Math.pow(line.weight, 2) + 1)
            var f = t + Math.pow(C, 2);
            if (f < 0) {
                //不存在交点
                //工程上应该允许有个误差值？误差多少合适？相切的点？
                //过圆心、垂直与直线的线段，线段的距离是否等于半径？
                return false;
            }
            var l = Math.sqrt(f)

            var ret = [];

            var x1 = l - C;
            var y1 = line.weight * x1 + line.bias;
            if (in_boundary(x1, y1)) {
                ret.push({ 'x': Math.round(x1), 'y': Math.round(y1) });
            }

            if (l != 0) {
                var x2 = - l - C;
                var y2 = line.weight * x2 + line.bias;
                if (in_boundary(x2, y2)) {
                    ret.push({ 'x': Math.round(x2), 'y': Math.round(y2) });
                }
            }
            return ret;
        }

        //计算圆和圆的相交点
        function calc_circle2_intersect_points(circle, circle1) {
            //1. 两个圆心连线的长度，大于两圆半径和->无交点，等于半径和->1个交点，小于半径和->2个交点
            //2. 如果有2个交点，则交点之间的连线垂直于圆心之间的连线，可计算交点连线的斜率
            //3. 如果有1个交点
            // 如果两个圆有交点，则交点的连线必然垂直于，两个圆心连接的直线 
            // 圆方程： 
            // (x-x0)²+(y-y0)²=r0²
            // (x-x1)²+(y-y1)²=r1²
            // (x-x1)²+(y-y1)² - ( (x-x0)² + (y-y0)² ) = r1²-r0²
            // (x² + x1² - 2*x1*x + y²+y1²-2*y1*y) - (x² + x0² - 2*x0*x + y²+y0²-2*y0*y) = r1²-r0²
            // x1² - 2*x1*x + y1²-2*y1*y - x0² + 2*x0*x - y0² + 2*y0*y = r1²-r0²
            // 2*x0*x - 2*x1*x + 2*y0*y -2*y1*y = r1²-r0² + y0² + x0² - y1² - x1²
            // (x0-x1)*x + (y0-y1)*y =  (r1²-r0² + y0² + x0² - y1² - x1²)/2 
            // y + ((x0-x1)/(y0-y1))*x =  (r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1)
            // y = - ((x0-x1)/(y0-y1))*x + (r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1)

            // y = Ax + B ? A=((x1-x0)/(y0-y1)) B=(r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1) 

            // x² + x0² - 2*x0*x + (Ax+B -y0)² = r0² 
            // x² - 2*x0*x + A²*x² + 2*(B-y0)*A*x  = r0² - x0² - (B-y0)²
            // (A²+1)*x² + 2*((B*A-y0*A-x0))*x = r0² - x0² - (B-y0)²
            // x² + 2* ((B*A-y0*A-x0)/(A²+1) )*x = (r0² - x0² - (B-y0)²)/(A²+1) 
            // error: (A²+1)*x² + 2*(B-y0-x0)*x = r0² - x0² - (B-y0)² 
            // C = ((B*A-y0*A-x0)/(A²+1))
            // x² + 2*C*x + C² = (r0² - x0² - (B-y0)²)/(A²+1) + C²  

            var A = (circle1.x - circle.x) / (circle.y - circle1.y);
            //B= (r1²-r0² + x0² - x1² + y0² - y1²)/2/(y0-y1)
            var B0 = Math.pow(circle1.radius, 2) - Math.pow(circle.radius, 2) + Math.pow(circle.x, 2) - Math.pow(circle1.x, 2) + Math.pow(circle.y, 2) - Math.pow(circle1.y, 2);
            var B = B0 / 2 / (circle.y - circle1.y);

            //C = ((B*A-y0*A-x0)/(A²+1))
            var C = ((B - circle.y) * A - circle.x) / (Math.pow(A, 2) + 1);
            // d0 = (r0² - x0² - (B-y0)²)/(A²+1) 
            var d0 = (Math.pow(circle.radius, 2) - Math.pow(circle.x, 2) - Math.pow(B - circle.y, 2)) / (Math.pow(A, 2) + 1)
            var d1 = d0 + Math.pow(C, 2);
            if (d1 < 0) {
                return false;
            }

            var ret = [];
            var D = Math.sqrt(d1);

            var x = - D - C;
            var y = A * x + B
            if (in_boundary(x, y)) {
                ret.push({ 'x': Math.round(x), 'y': Math.round(y) });
            }

            if (d1 != 0) {
                var x1 = D - C;
                var y1 = A * x1 + B;
                if (in_boundary(x1, y1)) {
                    ret.push({ 'x': Math.round(x1), 'y': Math.round(y1) });
                }
            }

            return ret;

        }

        //过圆心且垂直与线段的直线
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
            var radius = calc_distance(circle);

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

        //计算给定一点与所有相交点之间的距离
        function get_nearest_point(x, y, points) {
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

        //获取所有的相交点
        function get_intersect_points(include_last=true) {
            var tmp_points = [];

            //计算两个图形相交的点
            var len = operation_list.length;
            if (len < 2) {
                return tmp_points;
            }
            if(!include_last){
                len = len - 1 ;
            }
            for (var i = 0; i < len; i++) {
                for (var j = i + 1; j < len; j++) {
                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.LINE) {
                        //两直线相交
                        var points = calc_line2_intersect_point(operation_list[i], operation_list[j]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }
                    }

                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.CIRCLE) {
                        //直线&圆
                        var points = calc_line_circle_intersect_points(operation_list[i], operation_list[j]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }

                        // var points = calc_chuizhidian(operation_list[i], operation_list[j]);
                        // if (points) {
                        //     tmp_points.push.apply(tmp_points, points);
                        // }

                    }

                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.LINE) {
                        //圆&直线
                        var points = calc_line_circle_intersect_points(operation_list[j], operation_list[i]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }
                        // var points = calc_chuizhidian(operation_list[j], operation_list[i]);
                        // if (points) {
                        //     tmp_points.push.apply(tmp_points, points);
                        // }

                    }
                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.CIRCLE) {
                        //圆&圆
                        var points = calc_circle2_intersect_points(operation_list[i], operation_list[j]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }
                    }
                }

            }

            //用户绘制点+相交点去重
            var intersect_points = [];
            var manual_points = get_manual_points(false);
            for (var i = 0; i < tmp_points.length; i++) {
                var dm = get_nearest_point(tmp_points[i].x, tmp_points[i].y, manual_points);
                var d = get_nearest_point(tmp_points[i].x, tmp_points[i].y, intersect_points);
                if (!d && !dm) {
                    intersect_points.push({ 'x': tmp_points[i].x, 'y': tmp_points[i].y });
                }
            }
            return intersect_points;
        }


        //人工绘制的点
        function get_manual_points(include_last=true) {
            var tmp = [];
            var len_operation_list = operation_list.length;
            if(!include_last){
                len_operation_list = len_operation_list - 1 ;
            }

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
                var d = get_nearest_point(tmp[i].x, tmp[i].y, manual_points);
                if (!d) {
                    manual_points.push({ 'x': tmp[i].x, 'y': tmp[i].y });
                }
            } 

            return manual_points;
        }

        function has_same_objects(last_item) {
            //判断是否存在重复的绘图对象 
            var len_operation_list = operation_list.length;
            for (var i = 0; i < len_operation_list; i++) {
                var item = operation_list[i];

                //2点距离接近一个值，不需要
                // if (last_item.type == Tools.POINT) { 
                // }

                //2条线(w,b的差值)，
                if (last_item.type == Tools.LINE) {
                    if (Math.abs(last_item.weight - item.weight) < 0.01 ||
                        Math.abs(last_item.bias - item.bias) < 2) {
                        return true;
                    }
                }

                //2个圆(radius)
                if (last_item.type == Tools.CIRCLE) {
                    if (Math.abs(last_item.radius - item.radius) < 1) {
                        return true;
                    }
                }
            }

            return false;
        }

        //数据处理，类似mvc里的controler
        function process_data() {
            var len_operation_list = operation_list.length; 
            if (len_operation_list < 1) {
                return false;
            }

            var last_index = len_operation_list - 1; 
            var last_item = operation_list[last_index];

            for (var i = 0; i < len_operation_list; i++) {
                var item = operation_list[i];

                //移动画布
                if (current_tool == Tools.HAND) {
                    //item的数据结构要改改？
                    item.x = item.x + movie_points.x_movie;
                    item.y = item.y + movie_points.y_movie;
                    item.x1 = item.x1 + movie_points.x_movie;
                    item.y1 = item.y1 + movie_points.y_movie;
                }

                //计算直线的斜率和偏置项，左侧边界点、右侧边界点等
                if (item.type == Tools.LINE) {
                    var ret = calc_line_parameters(item);
                    for (var k in ret) {
                        item[k] = ret[k];
                    }
                }

                //计算圆的半径 
                if (item.type == Tools.CIRCLE) {
                    item.radius = calc_distance(item);
                }

                //高亮
                item.highlight = false; 
                if (isDrawing ) {
                    if (item.type == Tools.LINE && current_tool != Tools.HAND) {
                        // y = ax+b
                        var y = item.weight * last_item.x1 + item.bias;
                        if (Math.abs(y - last_item.y1) < 2) {
                            item.highlight = true; 
                        } 
                    }

                    if (item.type == Tools.CIRCLE) {
                        // (x-x0)²+(y-y0)²=r² 
                        var r = Math.sqrt(Math.pow(last_item.x1 - item.x, 2) + Math.pow(last_item.y1 - item.y, 2));
                        if (Math.abs(r - item.radius) < 2) {
                            item.highlight = true;
                        }
                    } 
                } 
            } 

            //磁吸线？

            //磁吸点
            var points = get_manual_points(false);
            points.push.apply(points, get_intersect_points(false)); 

            // 判断起始点
            if(isDrawing && current_tool!=Tools.HAND){
                var nearest_point = get_nearest_point(last_item.x, last_item.y, points);
                if (nearest_point) {
                    last_item.x = nearest_point.x;
                    last_item.y = nearest_point.y;
                }

                // 判断终点
                if (last_item.type == Tools.LINE || last_item.type == Tools.CIRCLE) {
                    var nearest_point = get_nearest_point(last_item.x1, last_item.y1, points);
                    if (nearest_point) {
                        last_item.x1 = nearest_point.x;
                        last_item.y1 = nearest_point.y;
                    } 
                }
            }
        } 


        /// VIEW RENDER
        //画点
        function draw_point(item, strokeStyle, lineWidth) {

            ctx.beginPath();
            // ctx.lineWidth = lineWidth;
            if (item.highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
            }
            ctx.arc(item.x, item.y, lineWidth, 0, Math.PI * 2, true);
            ctx.closePath();
            ctx.stroke();
        }

        //绘制直线
        function draw_line(start, end, strokeStyle, lineWidth) {

            ctx.beginPath();
            ctx.lineWidth = lineWidth;
            if (start.highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
            }
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();
            ctx.closePath();
        }

        //绘制圆
        function draw_arc(item, strokeStyle) {
            ctx.beginPath();
            if (item.highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
            }
            ctx.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
            ctx.closePath();
            ctx.stroke();
        }

        function render() {
            //设置canvas大小
            canvas.height = document.documentElement.clientHeight;
            canvas.width = document.documentElement.clientWidth;

            //根据操作记录和撤销记录设置undo/redo按钮是否可用 


            //处理所有的线、点
            process_data();

            canvas_rect = canvas.getBoundingClientRect();
            ctx.clearRect(0, 0, canvas_rect.width, canvas_rect.height); //TODO  

            //绘制人工点
            var manual_points = get_manual_points();
            for (var item of manual_points) {
                draw_point(item, NORMAL_COLOR, 2);
            }

            //绘制交点
            var intersect_points = get_intersect_points();
            for (var item of intersect_points) {
                draw_point(item, DASHES_COLOR, 1.5);
            }

            for (var item of operation_list) {
                //线
                if (item.type == Tools.LINE) {
                    //绘制直线 
                    draw_line(item.left, item.right, DASHES_COLOR, 0.3);
                    //绘制两点之间的线段部分
                    draw_line(item, { 'x': item.x1, 'y': item.y1 }, NORMAL_COLOR, 1);
                }

                //绘制圆
                if (item.type == Tools.CIRCLE) {
                    draw_arc(item, NORMAL_COLOR);
                }
            }

        }

        //事件绑定
        function event_binding() {
            //绘图工具框 
            toolbox.addEventListener('click', function () {
                var e = event || window.event;
                if (e.target && e.target.nodeName.toUpperCase() == "INPUT") {
                    current_tool = e.target.value;
                }
            }, false);

            //undo
            btn_undo.addEventListener('click', function () {
                var last = operation_list.pop();
                if (last) {
                    undo_operation_list.push(last);
                    render();
                }
            });

            //redo 
            btn_redo.addEventListener('click', function () {
                var last = undo_operation_list.pop();
                if (last) {
                    operation_list.push(last);
                    render();
                }
            });

            //两种习惯：暂时先支持第一种
            //1. 拖拽式 
            //2. 点击式 1.鼠标点击2下确定一条直线，一个圆等  
            canvas.addEventListener('mousedown', function (e) {
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                isDrawing = true;

                if (current_tool == Tools.HAND) { 
                    movie_points.x = x;
                    movie_points.y = y; 
                } else {
                    var obj = { "type": current_tool, 'x': x, 'y': y };
                    operation_list.push(obj);
                }

                requestAnimationFrame_status = window.requestAnimationFrame(render);
            });
            canvas.addEventListener('mousemove', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return false;
                }
                if (!isDrawing) {
                    return false;
                }

                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                if (current_tool == Tools.HAND) {
                    //移动画布
                    movie_points.x_movie = x - movie_points.x;
                    movie_points.y_movie = y - movie_points.y;
                    movie_points.x = x;
                    movie_points.y = y; 
                } else {
                    operation_list[last_index].x1 = x;
                    operation_list[last_index].y1 = y;
                }

                requestAnimationFrame_status = window.requestAnimationFrame(render);

            });
            canvas.addEventListener('mouseup', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0) {
                    return false;
                }

                if (isDrawing) { 
                    undo_operation_list = []; 
                }
                isDrawing = false;  
                
                window.cancelAnimationFrame(requestAnimationFrame_status);
                render();
            });

            canvas.addEventListener('click', function (e) {
                //
            });
            //右键单击、取消？

            //浏览器大小调整
            window.addEventListener('resize', function (e) {
                requestAnimationFrame_status = window.requestAnimationFrame(render);
            });

            window.addEventListener('load', function (e) {
                //页面加载，load用户的历史操作记录，呈现给用户？
            });
        }


        event_binding();
    }

    main();

})();